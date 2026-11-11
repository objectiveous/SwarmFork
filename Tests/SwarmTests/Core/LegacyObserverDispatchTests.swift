import Foundation
@testable import Swarm
import Testing

private actor LegacyOnlyCompatibilityObserver: AgentObserver {
    private(set) var received: [[MemoryMessage]] = []

    func onLLMStart(context: AgentContext?, agent: any AgentRuntime, systemPrompt: String?, inputMessages: [MemoryMessage]) async {
        received.append(inputMessages)
    }
}

private actor ModernOnlyCompatibilityObserver: AgentObserver {
    private(set) var received: [[InferenceMessage]] = []

    func onLLMStart(context: AgentContext?, agent: any AgentRuntime, systemPrompt: String?, inputMessages: [InferenceMessage]) async {
        received.append(inputMessages)
    }
}

private actor DualCompatibilityObserver: AgentObserver {
    private(set) var legacy: [[MemoryMessage]] = []
    private(set) var modern: [[InferenceMessage]] = []

    func onLLMStart(context: AgentContext?, agent: any AgentRuntime, systemPrompt: String?, inputMessages: [MemoryMessage]) async {
        legacy.append(inputMessages)
    }

    func onLLMStart(context: AgentContext?, agent: any AgentRuntime, systemPrompt: String?, inputMessages: [InferenceMessage]) async {
        modern.append(inputMessages)
    }
}

@Suite("Observer compatibility dispatch")
struct LegacyObserverDispatchTests {
    // Construct the real agent/provider, but test hooks directly: no inference,
    // networking, fake model, or agent run is needed for witness dispatch.
    private func agent() throws -> Agent {
        try Agent(
            tools: [],
            instructions: "Observer dispatch qualification.",
            memory: ConversationMemory(maxMessages: 10),
            inferenceProvider: .openAICompatible(
                baseURL: URL(string: "http://127.0.0.1:1/v1")!,
                model: "unused-for-observer-dispatch"
            )
        )
    }

    @Test("A legacy-only observer receives the encoded transcript once")
    func legacyOnly() async throws {
        let observer = LegacyOnlyCompatibilityObserver()
        let messages: [InferenceMessage] = [.system("System"), .user("Question")]
        await observer.notifyLLMStart(context: nil, agent: try agent(), systemPrompt: "System", inputMessages: messages)
        let received = await observer.received
        #expect(received.count == 1)
        #expect(received.first?.map(\.content) == messages.map(\.content))
        #expect(received.first?.map { $0.role.rawValue } == messages.map { $0.role.rawValue })
    }

    @Test("A modern-only observer receives the original transcript once")
    func modernOnly() async throws {
        let observer = ModernOnlyCompatibilityObserver()
        let messages: [InferenceMessage] = [.user("Question")]
        await observer.notifyLLMStart(context: nil, agent: try agent(), systemPrompt: nil, inputMessages: messages)
        #expect(await observer.received == [messages])
    }

    @Test("An observer implementing both overloads receives each once")
    func dual() async throws {
        let observer = DualCompatibilityObserver()
        let messages: [InferenceMessage] = [.user("Question"), .assistant("Answer")]
        await observer.notifyLLMStart(context: nil, agent: try agent(), systemPrompt: nil, inputMessages: messages)
        #expect(await observer.modern == [messages])
        let legacy = await observer.legacy
        #expect(legacy.count == 1)
        #expect(legacy.first?.map(\.content) == messages.map(\.content))
        #expect(legacy.first?.map { $0.role.rawValue } == messages.map { $0.role.rawValue })
    }
}
