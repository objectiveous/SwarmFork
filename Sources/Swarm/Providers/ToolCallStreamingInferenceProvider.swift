// ToolCallStreamingInferenceProvider.swift
// Swarm Framework
//
// Streaming tool-call updates on the InferenceProvider seam.

import Foundation

/// Provider-originated streaming updates used for live tool-call experiences.
public enum InferenceStreamUpdate: Sendable, Equatable {
    /// A chunk of assistant text produced during streaming.
    case outputChunk(String)

    /// A partial tool call update (arguments JSON fragment).
    case toolCallPartial(PartialToolCallUpdate)

    /// Completed tool calls ready for execution.
    case toolCallsCompleted([InferenceResponse.ParsedToolCall])

    /// Token usage statistics (typically available at the end of streaming).
    case usage(TokenUsage)

    /// A finished provider-owned turn, including the inner transcript.
    case finishedTurn(InferenceResponse)

    /// A fragment of the model's thinking text (a readable summary on
    /// Anthropic models), streamed before the answer or tool calls.
    ///
    /// Display only. The finished, signed block arrives as
    /// ``thinkingCompleted(_:)``; do not join fragments to rebuild it.
    case thinkingChunk(String)

    /// A finished thinking block, with the provider's signature when it has one.
    ///
    /// Agent keeps these on the assistant message so the provider can send
    /// them back unchanged on the next request of the same turn (Anthropic
    /// requires this on tool-use rounds when thinking is on).
    case thinkingCompleted(InferenceReasoning)
}

/// One finished block of model thinking that a provider may need back.
///
/// Providers that sign their thinking (Anthropic) must receive the block
/// unchanged on later requests of the same conversation. Treat every field as
/// opaque provider data: never edit ``text`` or ``signature``.
public struct InferenceReasoning: Sendable, Equatable, Codable {
    /// Thinking text (a summary on Anthropic models); `""` when the provider
    /// omitted it, for example under Anthropic's `display: "omitted"`.
    public var text: String

    /// Opaque signature the provider requires back unchanged. For Anthropic
    /// `redacted_thinking` blocks this carries the block's opaque data.
    public var signature: String?

    /// Provider format tag, for example `"anthropic-claude-v1"`.
    public var format: String?

    /// Provider block kind, for example `"reasoning.summary"` (thinking) or
    /// `"reasoning.encrypted"` (Anthropic redacted thinking).
    public var type: String?

    /// Creates a reasoning block.
    public init(
        text: String,
        signature: String? = nil,
        format: String? = nil,
        type: String? = nil
    ) {
        self.text = text
        self.signature = signature
        self.format = format
        self.type = type
    }
}

/// An inference provider that can stream tool-call assembly.
///
/// This deprecated protocol remains for source compatibility. Agent dispatches
/// live streaming from ``InferenceProviderCapabilities/streamingToolCalls``
/// and ``InferenceProvider/streamWithToolCalls(messages:tools:options:)``.
@available(*, deprecated, message: "Declare InferenceProviderCapabilities.streamingToolCalls and override streamWithToolCalls on InferenceProvider")
public protocol ToolCallStreamingInferenceProvider: InferenceProvider {
    /// Streams a response with potential tool calls, yielding partial tool-call updates as they arrive.
    func streamWithToolCalls(
        prompt: String,
        tools: [ToolSchema],
        options: InferenceOptions
    ) -> AsyncThrowingStream<InferenceStreamUpdate, Error>
}
