import Foundation
@testable import Swarm
import Testing

/// A session and a memory that keeps the conversation itself both hold the same turns. The model
/// must read each earlier turn once: from the session's history, not again from the memory's
/// context in the system message.
@Suite("Session history is sent once with a conversation memory", .ephemeralDefaultStores)
struct SessionMemoryDoubleFeedTests {
    @Test("PersistentMemory with a session sends each earlier turn once")
    func persistentMemoryWithSessionSendsHistoryOnce() async throws {
        let memory = PersistentMemory(backend: InMemoryBackend(), conversationId: "double-feed")
        let calls = try await runThreeTurns(memory: memory, session: InMemorySession(sessionId: "double-feed"))

        #expect(occurrences(of: "first-question-marker", in: calls[1]) == 1)
        #expect(occurrences(of: "first-answer-marker", in: calls[1]) == 1)
        #expect(occurrences(of: "first-question-marker", in: calls[2]) == 1)
        #expect(occurrences(of: "second-question-marker", in: calls[2]) == 1)
    }

    @Test("ConversationMemory with a session sends each earlier turn once")
    func conversationMemoryWithSessionSendsHistoryOnce() async throws {
        let calls = try await runThreeTurns(
            memory: ConversationMemory(maxMessages: 100),
            session: InMemorySession(sessionId: "double-feed-conversation")
        )

        #expect(occurrences(of: "first-question-marker", in: calls[1]) == 1)
        #expect(occurrences(of: "first-question-marker", in: calls[2]) == 1)
        #expect(occurrences(of: "second-question-marker", in: calls[2]) == 1)
    }

    @Test("A stored conversation memory from an earlier run is not replayed beside the session")
    func storedConversationMemoryIsNotReplayedBesideSession() async throws {
        let memory = PersistentMemory(backend: InMemoryBackend(), conversationId: "stale")
        await memory.add(.user("first-question-marker"))
        await memory.add(.assistant("first-answer-marker"))
        let session = InMemorySession(sessionId: "stale")
        try await session.addItems([.user("first-question-marker"), .assistant("first-answer-marker")])
        let provider = MockInferenceProvider(responses: ["second-answer-marker"])
        let agent = try Agent(
            "You are helpful.",
            configuration: AgentConfiguration(name: "stale", defaultTracingEnabled: false),
            memory: memory,
            inferenceProvider: provider
        )

        _ = try await agent.run("second-question-marker", session: session)

        let messages = try #require(await provider.generateMessageCalls.first?.messages)
        #expect(occurrences(of: "first-question-marker", in: messages) == 1)
    }

    private func runThreeTurns(memory: any Memory, session: any Session) async throws -> [[InferenceMessage]] {
        let provider = MockInferenceProvider(
            responses: ["first-answer-marker", "second-answer-marker", "third-answer-marker"]
        )
        let agent = try Agent(
            "You are helpful.",
            configuration: AgentConfiguration(name: "double-feed", defaultTracingEnabled: false),
            memory: memory,
            inferenceProvider: provider
        )

        _ = try await agent.run("first-question-marker", session: session)
        _ = try await agent.run("second-question-marker", session: session)
        _ = try await agent.run("third-question-marker", session: session)

        let calls = await provider.generateMessageCalls.map(\.messages)
        try #require(calls.count == 3)
        return calls
    }

    private func occurrences(of marker: String, in messages: [InferenceMessage]) -> Int {
        messages.reduce(0) { total, message in
            total + message.content.components(separatedBy: marker).count - 1
        }
    }
}
