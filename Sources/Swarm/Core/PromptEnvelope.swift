import Foundation

/// Package-internal context-window decisions shared by ``PromptEnvelope`` and
/// ``SlidingWindowMemory``.
///
/// Callers inject a token counter so the same fit/truncate policy can run
/// against a provider counter, ``EstimatedPromptTokenCounter``, or a test fake.
/// Roles stay; this is windowing, not flattening to a single user string.
enum ContextWindow {
    struct Policy: Equatable, Sendable {
        let maxTokens: Int
        let protectLeadingSystem: Bool
        let alwaysKeepLast: Bool

        static func fitting(_ profile: ContextProfile) -> Policy {
            Policy(
                maxTokens: profile.budget.maxInputTokens,
                protectLeadingSystem: true,
                alwaysKeepLast: true
            )
        }

        static func strict4k(_ profile: ContextProfile) -> Policy {
            fitting(profile)
        }
    }

    /// Drops oldest unprotected messages until the conversation fits `policy.maxTokens`.
    ///
    /// Token counts are measured on ``InferenceMessage/flattenPrompt(_:)``. The injected
    /// counter receives that labeled string, not raw message bodies.
    ///
    /// Messages leave as exchanges (see ``exchanges(in:)``): an assistant message with
    /// tool calls and the tool results that follow it are kept or dropped together, so a
    /// result never survives without its call, nor a call without its results.
    ///
    /// When `alwaysKeepLast` is true, the latest exchange is kept, and so is the latest
    /// user message before it: a question survives the lookups made to answer it. A
    /// leading `.system` message is kept when `protectLeadingSystem` is true; if those
    /// still overflow, their text is truncated from the newest message backward (and if
    /// needed the system text) so a non-empty system remains.
    ///
    /// When `alwaysKeepLast` is false, eviction is drop-oldest only (no truncation).
    /// At least one exchange is always retained.
    static func fit(
        messages: [InferenceMessage],
        policy: Policy,
        countTokens: @Sendable (String) async -> Int
    ) async -> [InferenceMessage] {
        guard !messages.isEmpty else {
            return messages
        }

        let maxTokens = policy.maxTokens
        if await tokenCount(of: messages, countTokens: countTokens) <= maxTokens {
            return messages
        }

        let hasProtectedSystem = policy.protectLeadingSystem && messages[0].role == .system
        let protectedSystem = hasProtectedSystem ? messages[0] : nil
        let units = exchanges(in: Array(messages.dropFirst(hasProtectedSystem ? 1 : 0)))

        if !policy.alwaysKeepLast {
            return await dropOldest(
                units,
                protectedSystem: protectedSystem,
                maxTokens: maxTokens,
                countTokens: countTokens
            )
        }

        guard let tailIndex = units.indices.last else {
            // The protected system is the only message.
            return [await truncatedMessage(
                messages[0],
                prefixedBy: [],
                maxTokens: maxTokens,
                keepNonEmpty: true,
                countTokens: countTokens
            )]
        }

        var pinned: Set<Int> = [tailIndex]
        if units[tailIndex].first?.role != .user,
           let questionIndex = units.lastIndex(where: { $0.first?.role == .user })
        {
            pinned.insert(questionIndex)
        }

        var kept = pinned
        for index in units.indices.reversed() where !pinned.contains(index) {
            let candidate = kept.union([index])
            let next = assemble(protectedSystem: protectedSystem, units: units, keeping: candidate)
            if await tokenCount(of: next, countTokens: countTokens) <= maxTokens {
                kept = candidate
            } else {
                break
            }
        }

        let windowed = assemble(protectedSystem: protectedSystem, units: units, keeping: kept)
        if await tokenCount(of: windowed, countTokens: countTokens) <= maxTokens {
            return windowed
        }

        return await fitProtectedSystemAndTail(
            protectedSystem,
            tail: units.indices.filter { pinned.contains($0) }.flatMap { units[$0] },
            maxTokens: maxTokens,
            countTokens: countTokens
        )
    }

    /// Splits messages into the units that are kept or dropped whole: an assistant
    /// message with tool calls together with the tool results that directly follow it,
    /// and every other message on its own. A tool result with no call before it stands
    /// alone.
    static func exchanges(in messages: [InferenceMessage]) -> [[InferenceMessage]] {
        var units: [[InferenceMessage]] = []
        for message in messages {
            if message.role == .tool,
               let opener = units.last?.first,
               opener.role == .assistant,
               !opener.toolCalls.isEmpty
            {
                units[units.count - 1].append(message)
            } else {
                units.append([message])
            }
        }
        return units
    }

    /// Longest character prefix of `text` whose token count is within `maxTokens`.
    ///
    /// If even `minimum` overflows, `minimum` is returned so callers can keep a
    /// non-empty protected system head.
    static func longestPrefix(
        of text: String,
        maxTokens: Int,
        minimum: String = "",
        countTokens: @Sendable (String) async -> Int
    ) async -> String {
        guard maxTokens > 0 else {
            return minimum
        }

        if await countTokens(text) <= maxTokens {
            return text
        }

        var lower = minimum.count
        var upper = text.count
        var best = minimum

        while lower <= upper {
            let mid = (lower + upper) / 2
            let candidate = String(text.prefix(mid))
            if await countTokens(candidate) <= maxTokens {
                best = candidate
                lower = mid + 1
            } else {
                upper = mid - 1
            }
        }

        return best
    }

    /// Synchronous counterpart of ``longestPrefix(of:maxTokens:minimum:countTokens:)``
    /// for callers that cannot suspend (actor-isolated mutation).
    static func longestPrefix(
        of text: String,
        maxTokens: Int,
        minimum: String = "",
        countTokens: (String) -> Int
    ) -> String {
        guard maxTokens > 0 else {
            return minimum
        }

        if countTokens(text) <= maxTokens {
            return text
        }

        var lower = minimum.count
        var upper = text.count
        var best = minimum

        while lower <= upper {
            let mid = (lower + upper) / 2
            let candidate = String(text.prefix(mid))
            if countTokens(candidate) <= maxTokens {
                best = candidate
                lower = mid + 1
            } else {
                upper = mid - 1
            }
        }

        return best
    }

    /// Keep-newest eviction: drop oldest messages until the additive token sum
    /// fits, always leaving at least one message.
    static func evictOldest<Message: Sendable>(
        from messages: [Message],
        maxTokens: Int,
        tokensOf: @Sendable (Message) async -> Int
    ) async -> [Message] {
        guard !messages.isEmpty else {
            return messages
        }

        var tokenCounts: [Int] = []
        tokenCounts.reserveCapacity(messages.count)
        for message in messages {
            tokenCounts.append(await tokensOf(message))
        }
        return evictOldest(from: messages, maxTokens: maxTokens, tokenCounts: tokenCounts)
    }

    /// Synchronous counterpart of ``evictOldest(from:maxTokens:tokensOf:)`` for
    /// callers that cannot suspend (actor-isolated mutation).
    static func evictOldest<Message>(
        from messages: [Message],
        maxTokens: Int,
        tokensOf: (Message) -> Int
    ) -> [Message] {
        evictOldest(
            from: messages,
            maxTokens: maxTokens,
            tokenCounts: messages.map(tokensOf)
        )
    }

    private static func evictOldest<Message>(
        from messages: [Message],
        maxTokens: Int,
        tokenCounts: [Int]
    ) -> [Message] {
        guard !messages.isEmpty else {
            return messages
        }

        var remaining = messages
        var counts = tokenCounts
        var total = counts.reduce(0, +)
        while total > maxTokens, remaining.count > 1 {
            remaining.removeFirst()
            total -= counts.removeFirst()
        }
        return remaining
    }

    private static func dropOldest(
        _ units: [[InferenceMessage]],
        protectedSystem: InferenceMessage?,
        maxTokens: Int,
        countTokens: @Sendable (String) async -> Int
    ) async -> [InferenceMessage] {
        var remaining = units
        let floorCount = protectedSystem == nil ? 1 : 0
        func joined() -> [InferenceMessage] {
            (protectedSystem.map { [$0] } ?? []) + remaining.flatMap { $0 }
        }
        while remaining.count > floorCount,
            await tokenCount(of: joined(), countTokens: countTokens) > maxTokens
        {
            remaining.removeFirst()
        }
        return joined()
    }

    private static func tokenCount(
        of messages: [InferenceMessage],
        countTokens: @Sendable (String) async -> Int
    ) async -> Int {
        await countTokens(InferenceMessage.flattenPrompt(messages))
    }

    private static func assemble(
        protectedSystem: InferenceMessage?,
        units: [[InferenceMessage]],
        keeping kept: Set<Int>
    ) -> [InferenceMessage] {
        let older = units.indices.filter { kept.contains($0) }.flatMap { units[$0] }
        return (protectedSystem.map { [$0] } ?? []) + older
    }

    /// Fits a protected leading system and the kept tail (the latest exchange and the
    /// question before it) under `maxTokens`. Prefers a full system + truncated tail
    /// (head-preserving). Only clips system text when the system message itself cannot
    /// share the budget.
    private static func fitProtectedSystemAndTail(
        _ protectedSystem: InferenceMessage?,
        tail: [InferenceMessage],
        maxTokens: Int,
        countTokens: @Sendable (String) async -> Int
    ) async -> [InferenceMessage] {
        if let system = protectedSystem {
            let clippedTail = await truncatedTail(
                tail,
                prefixedBy: [system],
                maxTokens: maxTokens,
                countTokens: countTokens
            )
            let withFullSystem = [system] + clippedTail
            if await tokenCount(of: withFullSystem, countTokens: countTokens) <= maxTokens {
                return withFullSystem
            }

            // System itself overflows even with an empty tail. Keep a non-empty
            // system head and a tail, matching the flattened-prompt contract.
            let tailShare = max(16, maxTokens / 3)
            let headShare = max(16, maxTokens - tailShare)
            let clippedSystem = await truncatedMessage(
                system,
                prefixedBy: [],
                maxTokens: headShare,
                keepNonEmpty: true,
                countTokens: countTokens
            )
            return [clippedSystem] + (await truncatedTail(
                tail,
                prefixedBy: [clippedSystem],
                maxTokens: maxTokens,
                countTokens: countTokens
            ))
        }

        return await truncatedTail(tail, prefixedBy: [], maxTokens: maxTokens, countTokens: countTokens)
    }

    /// Truncates tail messages from the newest backward until `prefix + tail` fits,
    /// so tool results are cut before the question they answer.
    private static func truncatedTail(
        _ tail: [InferenceMessage],
        prefixedBy prefix: [InferenceMessage],
        maxTokens: Int,
        countTokens: @Sendable (String) async -> Int
    ) async -> [InferenceMessage] {
        var result = tail
        for index in result.indices.reversed() {
            if await tokenCount(of: prefix + result, countTokens: countTokens) <= maxTokens {
                break
            }
            result[index] = await truncatedMessage(
                result[index],
                prefixedBy: prefix + result[..<index],
                followedBy: Array(result[(index + 1)...]),
                maxTokens: maxTokens,
                keepNonEmpty: false,
                countTokens: countTokens
            )
        }
        return result
    }

    private static func truncatedMessage(
        _ message: InferenceMessage,
        prefixedBy prefix: [InferenceMessage],
        followedBy suffix: [InferenceMessage] = [],
        maxTokens: Int,
        keepNonEmpty: Bool,
        countTokens: @Sendable (String) async -> Int
    ) async -> InferenceMessage {
        let minimum = keepNonEmpty && !message.content.isEmpty
            ? String(message.content.prefix(1))
            : ""
        let clipped = await longestPrefix(
            of: message.content,
            maxTokens: maxTokens,
            minimum: minimum,
            countTokens: { candidate in
                let next = prefix + [replacingContent(of: message, with: candidate)] + suffix
                return await countTokens(InferenceMessage.flattenPrompt(next))
            }
        )
        return replacingContent(of: message, with: clipped)
    }

    private static func replacingContent(of message: InferenceMessage, with content: String) -> InferenceMessage {
        switch message.body {
        case .system:
            InferenceMessage(body: .system(content), attachments: message.attachments)
        case .user:
            InferenceMessage(body: .user(content), attachments: message.attachments)
        case let .assistant(_, toolCalls):
            InferenceMessage(
                body: .assistant(content, toolCalls: toolCalls),
                attachments: message.attachments
            )
        case let .tool(name, _, toolCallID):
            InferenceMessage(
                body: .tool(name: name, content: content, toolCallID: toolCallID),
                attachments: message.attachments
            )
        }
    }
}

/// Enforces context-envelope limits for provider prompts.
enum PromptEnvelope {
    static let omittedToolResult = "Tool result omitted to fit context."

    /// Drops oldest non-system messages until the conversation fits the profile budget.
    /// Roles stay. A tool call and its results leave together. The latest exchange and
    /// the user message it answers are always kept. A leading `.system` message is
    /// never dropped; if those still overflow, their text (and if needed system text)
    /// is truncated so a non-empty system remains.
    ///
    /// Over-budget tool bodies are stubbed first (all but the last two), then the
    /// shared ``ContextWindow`` core fits the remainder. This runs for every
    /// profile, not only ``ContextProfile/strict4k``.
    static func enforce(messages: [InferenceMessage], profile: ContextProfile) async -> [InferenceMessage] {
        guard !messages.isEmpty else {
            return messages
        }

        let budget = profile.budget.maxInputTokens
        var working = messages
        if await tokenCount(of: working) > budget {
            working = stubOlderToolResults(working, keepLast: 2)
        }

        return await ContextWindow.fit(
            messages: working,
            policy: .fitting(profile),
            countTokens: { await PromptTokenBudgeting.countTokens(in: $0) }
        )
    }

    /// Keeps a leading system message and the latest turn. Used after Apple
    /// rejects a prompt that still overflowed the estimated budget.
    static func compactForRetry(_ messages: [InferenceMessage]) -> [InferenceMessage] {
        guard let last = messages.last else {
            return messages
        }
        if messages.first?.role == .system, messages[0] != last {
            return [messages[0], last]
        }
        return [last]
    }

    private static func stubOlderToolResults(
        _ messages: [InferenceMessage],
        keepLast: Int
    ) -> [InferenceMessage] {
        let toolIndices = messages.indices.filter { messages[$0].role == .tool }
        guard toolIndices.count > keepLast else {
            return messages
        }
        let kept = Set(toolIndices.suffix(keepLast))
        return messages.enumerated().map { index, message in
            guard case let .tool(name, _, toolCallID) = message.body, kept.contains(index) == false else {
                return message
            }
            return InferenceMessage(
                body: .tool(name: name, content: omittedToolResult, toolCallID: toolCallID),
                attachments: message.attachments
            )
        }
    }

    private static func tokenCount(of messages: [InferenceMessage]) async -> Int {
        await PromptTokenBudgeting.countTokens(in: InferenceMessage.flattenPrompt(messages))
    }
}
