# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stub. Model-specific parser adapters are removed from the lean build."""

from vllm.parser.engine.adapters import make_adapters


class _StubParser:
    pass


DeepSeekV32Parser = _StubParser
DeepSeekV4Parser = _StubParser
Gemma4Parser = _StubParser
Glm47MoeParser = _StubParser
InklingParser = _StubParser
KimiK2Parser = _StubParser
MinimaxM2Parser = _StubParser
MistralParser = _StubParser
NemotronV3Parser = _StubParser
Qwen3Parser = _StubParser
SeedOssParser = _StubParser

(
    DeepSeekV32ParserReasoningAdapter,
    DeepSeekV32ParserToolAdapter,
) = make_adapters(DeepSeekV32Parser)

(
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
) = make_adapters(DeepSeekV4Parser)

(
    MinimaxM2ParserReasoningAdapter,
    MinimaxM2ParserToolAdapter,
) = make_adapters(MinimaxM2Parser)

(
    Gemma4ParserReasoningAdapter,
    Gemma4ParserToolAdapter,
) = make_adapters(Gemma4Parser)

(
    NemotronV3ParserReasoningAdapter,
    NemotronV3ParserToolAdapter,
) = make_adapters(NemotronV3Parser)

(
    Qwen3ParserReasoningAdapter,
    Qwen3ParserToolAdapter,
) = make_adapters(Qwen3Parser)

(
    SeedOssParserReasoningAdapter,
    SeedOSSParserToolAdapter,
) = make_adapters(SeedOssParser)

(
    Glm47MoeParserReasoningAdapter,
    Glm47MoeParserToolAdapter,
) = make_adapters(Glm47MoeParser)

(
    KimiK2ParserReasoningAdapter,
    KimiK2ParserToolAdapter,
) = make_adapters(KimiK2Parser)

(
    InklingParserReasoningAdapter,
    InklingParserToolAdapter,
) = make_adapters(InklingParser)

(
    MistralParserReasoningAdapter,
    MistralParserToolAdapter,
) = make_adapters(MistralParser)
