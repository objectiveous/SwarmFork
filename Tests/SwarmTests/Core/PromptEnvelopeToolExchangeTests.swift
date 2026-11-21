import Foundation
@testable import Swarm
import Testing

/// Shortening a conversation for a model with a small window: a tool call and its results leave
/// together, the question survives its lookups, and a profile is sized to one model's window
/// with a margin for the character estimate.
@Suite("PromptEnvelope tool exchanges and model windows")
struct PromptEnvelopeToolExchangeTests {
    private let padding = String(repeating: "p", count: 400)

    // MARK: - A call and its results stay together

    @Test("an older call and its result are dropped together")
    func olderExchangeLeavesWhole() async {
        let messages: [InferenceMessage] = [
            .system("Stay."),
            .user("old-question \(padding)"),
            .assistant("", toolCalls: [call("a", "lookup")]),
            .tool(name: "lookup", content: "old-result \(padding)", toolCallID: "a"),
            .assistant("old-answer"),
            .user("needle-latest"),
        ]
        let budget = characters(in: [messages[0], messages[3], messages[5]])

        let fitted = await fit(messages, budget: budget)

        expectExchangesIntact(fitted)
        #expect(!fitted.contains { $0.content.contains("old-result") })
        #expect(fitted.last?.content == "needle-latest")
        #expect(characters(in: fitted) <= budget)
    }

    @Test("parallel calls and all their results stay or leave as one")
    func parallelCallsLeaveWhole() async {
        let messages: [InferenceMessage] = [
            .system("Stay."),
            .user("question"),
            .assistant("", toolCalls: [call("a", "first"), call("b", "second")]),
            .tool(name: "first", content: "first-result \(padding)", toolCallID: "a"),
            .tool(name: "second", content: "second-result", toolCallID: "b"),
            .assistant("answer"),
            .user("needle-latest"),
        ]
        // Room for the second result alone, never for the whole exchange.
        let budget = characters(in: [messages[0], messages[2], messages[4], messages[5], messages[6]])

        let fitted = await fit(messages, budget: budget)

        expectExchangesIntact(fitted)
        #expect(!fitted.contains { $0.content.contains("second-result") })
        #expect(fitted.last?.content == "needle-latest")
    }

    @Test("a result that ends the conversation keeps its call and the question it answers")
    func latestResultKeepsCallAndQuestion() async {
        // The one-4el7.12 case: after a lookup, the last message is the tool result.
        let messages: [InferenceMessage] = [
            .system("Stay."),
            .user("old-question \(padding)"),
            .assistant("old-answer \(padding)"),
            .user("needle-question"),
            .assistant("", toolCalls: [call("a", "lookup")]),
            .tool(name: "lookup", content: "fresh-result", toolCallID: "a"),
        ]
        let budget = characters(in: [messages[0]] + Array(messages[3...]))

        let fitted = await fit(messages, budget: budget)

        expectExchangesIntact(fitted)
        #expect(fitted.map(\.content) == ["Stay.", "needle-question", "", "fresh-result"])
    }

    @Test("lookups between the question and the latest exchange leave oldest first, whole")
    func earlierLookupsInTheTurnLeaveWhole() async {
        let messages: [InferenceMessage] = [
            .system("Stay."),
            .user("needle-question"),
            .assistant("", toolCalls: [call("a", "lookup")]),
            .tool(name: "lookup", content: "first-result \(padding)", toolCallID: "a"),
            .assistant("", toolCalls: [call("b", "lookup"), call("c", "lookup")]),
            .tool(name: "lookup", content: "second-result \(padding)", toolCallID: "b"),
            .tool(name: "lookup", content: "third-result", toolCallID: "c"),
            .assistant("", toolCalls: [call("d", "lookup")]),
            .tool(name: "lookup", content: "latest-result", toolCallID: "d"),
        ]
        let withoutFirst = [messages[0], messages[1]] + Array(messages[4...])
        let budget = characters(in: withoutFirst)

        let fitted = await fit(messages, budget: budget)

        expectExchangesIntact(fitted)
        #expect(fitted == withoutFirst)
    }

    @Test("an exchange too big for the window is cut from its newest result, not split")
    func oversizedExchangeIsTruncatedNotSplit() async {
        let messages: [InferenceMessage] = [
            .system("Stay."),
            .user("needle-question"),
            .assistant("", toolCalls: [call("a", "lookup"), call("b", "lookup")]),
            .tool(name: "lookup", content: "first-result", toolCallID: "a"),
            .tool(name: "lookup", content: "huge-result \(padding)\(padding)", toolCallID: "b"),
        ]
        let budget = characters(in: Array(messages[..<4])) + 120

        let fitted = await fit(messages, budget: budget)

        expectExchangesIntact(fitted)
        #expect(fitted.count == messages.count)
        #expect(fitted[1].content == "needle-question")
        #expect(fitted[3].content == "first-result")
        #expect(fitted[4].content.hasPrefix("huge-result"))
        #expect(fitted[4].content.count < messages[4].content.count)
        #expect(characters(in: fitted) <= budget)
    }

    @Test("drop-oldest without keeping the last also moves whole exchanges")
    func dropOldestMovesWholeExchanges() async {
        let messages: [InferenceMessage] = [
            .assistant("", toolCalls: [call("a", "lookup")]),
            .tool(name: "lookup", content: "old-result \(padding)", toolCallID: "a"),
            .user("needle-latest"),
        ]

        let fitted = await ContextWindow.fit(
            messages: messages,
            policy: ContextWindow.Policy(maxTokens: characters(in: [messages[1], messages[2]]), protectLeadingSystem: false, alwaysKeepLast: false),
            countTokens: { $0.count }
        )

        #expect(fitted == [messages[2]])
    }

    @Test("exchanges group a call with the results that follow it; a stray result stands alone")
    func exchangesGrouping() {
        let messages: [InferenceMessage] = [
            .tool(name: "lookup", content: "stray", toolCallID: "z"),
            .user("question"),
            .assistant("", toolCalls: [call("a", "lookup"), call("b", "lookup")]),
            .tool(name: "lookup", content: "a", toolCallID: "a"),
            .tool(name: "lookup", content: "b", toolCallID: "b"),
            .assistant("answer"),
        ]

        #expect(ContextWindow.exchanges(in: messages).map(\.count) == [1, 1, 3, 1])
    }

    // MARK: - A size limit for each model, with a margin for the estimate

    @Test("a model-window profile leaves room for the reply, the overhead and the margin")
    func modelWindowBudget() {
        let profile = ContextProfile.modelWindow(
            contextTokens: 32_768,
            outputReserveTokens: 4_096,
            protocolOverheadReserveTokens: 2_672,
            estimateMargin: 2
        )

        #expect(profile.budget.maxInputTokens == 13_000)
        #expect(profile.maxTotalContextTokens == 32_768)
        #expect(profile.outputReserveTokens == 4_096)
        #expect(profile.safetyMarginTokens == 13_000)
        #expect(ContextProfile.modelWindow(contextTokens: 8_192, outputReserveTokens: 1_024).budget.maxInputTokens == 7_168)
        #expect(ContextProfile.modelWindow(contextTokens: 100, outputReserveTokens: 500).budget.maxInputTokens == 1)
        #expect(ContextProfile.modelWindow(contextTokens: 1_000, outputReserveTokens: 0, estimateMargin: .nan).budget.maxInputTokens == 1_000)
    }

    @Test("each model's window trims the same conversation to its own size")
    func windowsTrimToTheirOwnSize() async {
        let conversation = longConversation(turns: 300)
        let small = ContextProfile.modelWindow(contextTokens: 4_096, outputReserveTokens: 1_024)
        let large = ContextProfile.modelWindow(contextTokens: 32_768, outputReserveTokens: 4_096)

        let smallFit = await PromptEnvelope.enforce(messages: conversation, profile: small)
        let largeFit = await PromptEnvelope.enforce(messages: conversation, profile: large)

        #expect(await estimatedTokens(smallFit) <= small.budget.maxInputTokens)
        #expect(await estimatedTokens(largeFit) <= large.budget.maxInputTokens)
        #expect(smallFit.count < largeFit.count)
        #expect(largeFit.count < conversation.count)
        expectExchangesIntact(smallFit)
        expectExchangesIntact(largeFit)
    }

    @Test("with the margin, a prompt whose real tokens run 1.92 times the estimate still fits the window")
    func marginCoversAnEstimateThatRunsLow() async {
        // 1.92 is the worst real-to-estimated ratio measured with Inkling 2 Mini's tokenizer
        // on Studio tool results; whole Studio sessions peaked at 1.72.
        let worstRatio = 1.92
        let window = 8_192
        let output = 1_024
        let conversation = longConversation(turns: 60)
        let withoutMargin = ContextProfile.modelWindow(contextTokens: window, outputReserveTokens: output)
        let withMargin = ContextProfile.modelWindow(contextTokens: window, outputReserveTokens: output, estimateMargin: 2)

        let unguarded = await PromptEnvelope.enforce(messages: conversation, profile: withoutMargin)
        let guarded = await PromptEnvelope.enforce(messages: conversation, profile: withMargin)

        let unguardedReal = Double(await estimatedTokens(unguarded)) * worstRatio
        let guardedReal = Double(await estimatedTokens(guarded)) * worstRatio
        #expect(unguardedReal > Double(window - output))
        #expect(guardedReal <= Double(window - output))
    }

    // MARK: - Helpers

    private func call(_ id: String, _ name: String) -> InferenceMessage.ToolCall {
        InferenceMessage.ToolCall(id: id, name: name, arguments: [:])
    }

    private func fit(_ messages: [InferenceMessage], budget: Int) async -> [InferenceMessage] {
        await ContextWindow.fit(
            messages: messages,
            policy: ContextWindow.Policy(maxTokens: budget, protectLeadingSystem: true, alwaysKeepLast: true),
            countTokens: { $0.count }
        )
    }

    private func characters(in messages: [InferenceMessage]) -> Int {
        InferenceMessage.flattenPrompt(messages).count
    }

    private func estimatedTokens(_ messages: [InferenceMessage]) async -> Int {
        await PromptTokenBudgeting.countTokens(in: InferenceMessage.flattenPrompt(messages))
    }

    /// A Studio-shaped conversation: each turn asks, looks something up, and answers.
    private func longConversation(turns: Int) -> [InferenceMessage] {
        var messages: [InferenceMessage] = [.system("You model what the user describes.")]
        for turn in 0 ..< turns {
            let id = "call-\(turn)"
            messages.append(.user("question \(turn): what links meetings to decisions?"))
            messages.append(.assistant("", toolCalls: [call(id, "inspect_studio_workbook")]))
            messages.append(.tool(
                name: "inspect_studio_workbook",
                content: "{\"item\":\"\(UUID().uuidString)\",\"turn\":\(turn),\"notes\":\"\(padding)\"}",
                toolCallID: id
            ))
            messages.append(.assistant("answer \(turn): a decision results from a meeting. \(padding)"))
        }
        return messages
    }

    /// Every tool result follows the assistant message that called it, and every call has its result.
    private func expectExchangesIntact(
        _ messages: [InferenceMessage],
        sourceLocation: Testing.SourceLocation = #_sourceLocation
    ) {
        for unit in ContextWindow.exchanges(in: messages) {
            guard let opener = unit.first else { continue }
            if opener.role == .tool {
                Issue.record("tool result \(opener.toolCallID ?? "?") has no call before it", sourceLocation: sourceLocation)
                continue
            }
            let calls = Set(opener.toolCalls.compactMap(\.id))
            let results = Set(unit.dropFirst().compactMap(\.toolCallID))
            #expect(calls == results, "calls \(calls) answered by \(results)", sourceLocation: sourceLocation)
        }
    }
}
