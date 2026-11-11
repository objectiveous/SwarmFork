@testable import Swarm
import Testing

@Suite("Provided context slot identity")
struct ContextSlotProvidedValueTests {
    @Test("Provided values preserve both concrete type and explicit key")
    func keysAndTypesRemainDistinct() {
        var store = ContextSlotStore()
        store.setProvided(7, contextKey: "shared")
        store.setProvided("seven", contextKey: "shared")
        store.setProvided(8, contextKey: "other")

        #expect(store.provided(of: Int.self, contextKey: "shared") == 7)
        #expect(store.provided(of: String.self, contextKey: "shared") == "seven")
        #expect(store.provided(of: Int.self, contextKey: "other") == 8)
        #expect(store.provided(of: Bool.self, contextKey: "shared") == nil)
        #expect(store.removeProvided(of: Int.self, contextKey: "shared") == 7)
        #expect(!store.containsProvided(of: Int.self, contextKey: "shared"))
        #expect(store.containsProvided(of: String.self, contextKey: "shared"))
        #expect(store.provided(of: Int.self, contextKey: "other") == 8)
    }

    @Test("Copying value slots continues to exclude legacy provided instances")
    func copyDoesNotDuplicateProvidedInstances() {
        var store = ContextSlotStore()
        store.setProvided(7, contextKey: "shared")
        let copy = store.copyingValueSlots()
        #expect(!copy.containsProvided(of: Int.self, contextKey: "shared"))
        #expect(store.containsProvided(of: Int.self, contextKey: "shared"))
    }
}
