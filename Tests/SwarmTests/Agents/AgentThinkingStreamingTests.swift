import Foundation
import Testing
@testable import Swarm

/// Thinking text and signed thinking blocks through the agent loop:
/// providers stream `thinkingChunk` / `thinkingCompleted`, the loop raises
/// thinking events, and the finished blocks ride the assistant message that
/// is sent back with the tool results on the next call.
@Suite("Agent thinking streaming", .ephemeralDefaultStores)
struct AgentThinkingStreamingTests {

    struct EchoTool: AnyJSONTool, Sendable {
        let name = "echo"
        let description = "Echoes the input text"
        let parameters: [ToolParameter] = [
            ToolParameter(name: "text", description: "Text to echo", type: .string)
        ]

        func execute(arguments: [String: SendableValue]) async throws -> SendableValue {
            .string(try requiredString("text", from: arguments))
        }
    }

    // Not an actor: `streamWithToolCalls` is a synchronous requirement (see
    // AgentLiveToolCallStreamingTests for the executor-assumption note).
    final class RecordingScriptedProvider: InferenceProvider, @unchecked Sendable {
        nonisolated let capabilities: InferenceProviderCapabilities

        private let lock = NSLock()
        private let scripts: [[InferenceStreamUpdate]]
        private let responses: [InferenceResponse]
        private var index = 0
        private var received: [[InferenceMessage]] = []

        init(
            capabilities: InferenceProviderCapabilities,
            scripts: [[InferenceStreamUpdate]] = [],
            responses: [InferenceResponse] = []
        ) {
            self.capabilities = capabilities
            self.scripts = scripts
            self.responses = responses
        }

        private func record(_ messages: [InferenceMessage]) -> Int {
            lock.lock()
            defer { lock.unlock() }
            received.append(messages)
            defer { index += 1 }
            return index
        }

        func receivedMessages() -> [[InferenceMessage]] {
            lock.lock()
            defer { lock.unlock() }
            return received
        }

        func generate(messages _: [InferenceMessage], options _: InferenceOptions) async throws -> String {
            throw AgentError.generationFailed(reason: "Unexpected call to generate() in thinking test")
        }

        func generateWithToolCalls(
            messages: [InferenceMessage],
            tools _: [ToolSchema],
            options _: InferenceOptions,
            toolExecutor _: ToolCallExecutor?
        ) async throws -> InferenceResponse {
            let call = record(messages)
            return responses[min(call, responses.count - 1)]
        }

        func streamWithToolCalls(
            messages: [InferenceMessage],
            tools _: [ToolSchema],
            options _: InferenceOptions,
            toolExecutor _: ToolCallExecutor?
        ) -> AsyncThrowingStream<InferenceStreamUpdate, Error> {
            let call = record(messages)
            let updates = scripts[min(call, scripts.count - 1)]
            return StreamHelper.makeTrackedStream { continuation in
                for update in updates {
                    continuation.yield(update)
                }
                continuation.finish()
            }
        }
    }

    static let signedBlock = InferenceReasoning(
        text: "I should echo the greeting.",
        signature: "sig-abc",
        format: "anthropic-claude-v1",
        type: "reasoning.summary"
    )

    static let echoCall = InferenceResponse.ParsedToolCall(
        id: "call_1",
        name: "echo",
        arguments: ["text": .string("hi")]
    )

    @Test("Streamed thinking raises partial and finished thinking events")
    func streamedThinkingRaisesEvents() async throws {
        let provider = RecordingScriptedProvider(
            capabilities: [.streamingToolCalls],
            scripts: [
                [
                    .thinkingChunk("I should echo"),
                    .thinkingChunk(" the greeting."),
                    .thinkingCompleted(Self.signedBlock),
                    .toolCallsCompleted([Self.echoCall]),
                ],
                [
                    .outputChunk("All done"),
                ],
            ]
        )
        let agent = try Agent(
            tools: [EchoTool()],
            configuration: .default.maxIterations(3),
            inferenceProvider: provider
        )

        var events: [AgentEvent] = []
        for try await event in agent.stream("Hi") {
            events.append(event)
        }

        let partials: [String] = events.compactMap { event in
            if case let .output(.thinkingPartial(text)) = event { return text }
            return nil
        }
        let thoughts: [String] = events.compactMap { event in
            if case let .output(.thinking(thought)) = event { return thought }
            return nil
        }
        #expect(partials == ["I should echo", " the greeting."])
        #expect(thoughts == ["I should echo the greeting."])

        let thoughtIndex = events.firstIndex { event in
            if case .output(.thinking) = event { return true }
            return false
        }
        let toolStartIndex = events.firstIndex { event in
            if case .tool(.started) = event { return true }
            return false
        }
        if let thoughtIndex, let toolStartIndex {
            #expect(thoughtIndex < toolStartIndex)
        } else {
            Issue.record("Missing thinking or tool start event")
        }
    }

    @Test("Signed thinking block is sent back on the assistant message before the tool result")
    func signedBlockReplayed() async throws {
        let provider = RecordingScriptedProvider(
            capabilities: [.streamingToolCalls],
            scripts: [
                [
                    .thinkingCompleted(Self.signedBlock),
                    .toolCallsCompleted([Self.echoCall]),
                ],
                [
                    .outputChunk("All done"),
                ],
            ]
        )
        let agent = try Agent(
            tools: [EchoTool()],
            configuration: .default.maxIterations(3),
            inferenceProvider: provider
        )
        for try await _ in agent.stream("Hi") {}

        let calls = provider.receivedMessages()
        #expect(calls.count == 2)
        let second = try #require(calls.last)
        let assistantIndex = try #require(second.lastIndex { !$0.toolCalls.isEmpty })
        let assistant = second[assistantIndex]
        #expect(assistant.role == .assistant)
        #expect(assistant.reasoning == [Self.signedBlock])
        #expect(assistant.toolCalls.first?.id == "call_1")
        let toolIndex = try #require(second.lastIndex { $0.role == .tool })
        #expect(assistantIndex < toolIndex)

        // Nothing else carries reasoning.
        #expect(second.filter { !$0.reasoning.isEmpty }.count == 1)
    }

    @Test("Non-streaming responses carry reasoning onto the assistant message")
    func nonStreamingReasoningReplayed() async throws {
        let provider = RecordingScriptedProvider(
            capabilities: [],
            responses: [
                InferenceResponse(
                    toolCalls: [Self.echoCall],
                    finishReason: .toolCall,
                    reasoning: [Self.signedBlock]
                ),
                InferenceResponse(content: "All done"),
            ]
        )
        let agent = try Agent(
            tools: [EchoTool()],
            configuration: .default.maxIterations(3),
            inferenceProvider: provider
        )
        let result = try await agent.run("Hi")
        #expect(result.output == "All done")

        let second = try #require(provider.receivedMessages().last)
        let assistant = try #require(second.last { !$0.toolCalls.isEmpty })
        #expect(assistant.reasoning == [Self.signedBlock])
    }

    @Test("A stream without thinking leaves messages without reasoning")
    func noThinkingUnchanged() async throws {
        let provider = RecordingScriptedProvider(
            capabilities: [.streamingToolCalls],
            scripts: [
                [.toolCallsCompleted([Self.echoCall])],
                [.outputChunk("All done")],
            ]
        )
        let agent = try Agent(
            tools: [EchoTool()],
            configuration: .default.maxIterations(3),
            inferenceProvider: provider
        )
        var events: [AgentEvent] = []
        for try await event in agent.stream("Hi") {
            events.append(event)
        }
        #expect(!events.contains { event in
            if case .output(.thinking) = event { return true }
            if case .output(.thinkingPartial) = event { return true }
            return false
        })
        let second = try #require(provider.receivedMessages().last)
        #expect(second.allSatisfy { $0.reasoning.isEmpty })
    }

    @Test("Finished non-streaming turns emit thinkingCompleted first")
    func finishedTurnEmitsThinkingFirst() async throws {
        let response = InferenceResponse(
            content: "Hello",
            reasoning: [Self.signedBlock]
        )
        var updates: [InferenceStreamUpdate] = []
        for try await update in streamFinishedToolTurn({ response }) {
            updates.append(update)
        }
        #expect(updates.first == .thinkingCompleted(Self.signedBlock))
        #expect(updates.dropFirst().first == .outputChunk("Hello"))
    }

    @Test("Joined thinking text skips empty blocks")
    func joinedThinkingText() {
        let text = AgentTurnRunner.joinedThinkingText([
            InferenceReasoning(text: "First."),
            InferenceReasoning(text: "", signature: "omitted"),
            InferenceReasoning(text: "Second."),
        ])
        #expect(text == "First.\n\nSecond.")
    }
}
