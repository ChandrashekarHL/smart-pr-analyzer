"""Security pass (design doc FR-07), built with LangChain.

    chain = build_chain(PranaChatModel(client))
    findings = review_security(chain, hunk, language, path)

The fine-tuned reviewer learned to write the single most typical human
comment, and humans rarely write about security (2.4% of the corpus), so it
under-detects vulnerabilities. This pass asks the *base* DeepSeek-Coder - the
same weights with the LoRA adapter switched off, so no extra GPU memory - one
narrow question: does this change introduce a vulnerability?

LangChain pieces, each doing real work:
- `ChatPromptTemplate` holds the security-review instruction and checklist;
- `PranaChatModel` is a `BaseChatModel` over our model API's `/generate`
  (adapter off), so any LangChain chat model could replace it;
- `JsonOutputParser` extracts the JSON (tolerating code fences), and Pydantic
  validates it into `Finding` objects.
A reply that does not parse yields no findings and a log line, never an error.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("prana_app.security_pass")

VULNERABILITY_CLASSES = (
    "SQL / command / code injection", "hard-coded secrets or credentials",
    "insecure deserialization", "missing validation of untrusted input",
    "path traversal", "cross-site scripting (XSS)", "server-side request forgery (SSRF)",
    "broken authentication or token verification", "weak cryptography or randomness",
    "disabled TLS certificate verification", "open redirect",
)

SYSTEM = (
    "You are an application-security reviewer. You look at ONE code change and report "
    "only real, exploitable security vulnerabilities that the change itself introduces. "
    "Check for: " + "; ".join(VULNERABILITY_CLASSES) + ". "
    "Do not report style, performance, missing tests or general bugs. If the code is "
    "safe, report no findings - an empty list is the correct answer for most changes.\n"
    'Answer with JSON only: {{"findings": [{{"vulnerability": "<class>", '
    '"line": <line number in the new code or null>, "severity": "critical" or "warning", '
    '"explanation": "<one sentence>", "fix": "<one sentence>"}}]}}'
)
HUMAN = "Language: {language}\nFile: {path}\n\n```diff\n{hunk}\n```"
STRICT = ('\n\nReply with the JSON object only - no prose, no markdown. If the change is '
          'safe, reply exactly {{"findings": []}}.')


class Finding(BaseModel):
    vulnerability: str = Field(min_length=3, max_length=120)
    line: int | None = None
    severity: Literal["critical", "warning"] = "warning"
    explanation: str = Field(min_length=10, max_length=600)
    fix: str = Field(default="", max_length=600)


class SecurityReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list, max_length=5)


class PranaChatModel(BaseChatModel):
    """LangChain chat model over the prana model API (`/generate`)."""

    client: Any
    use_adapter: bool = False            # base model: broad security knowledge
    max_new_tokens: int = 384

    @property
    def _llm_type(self) -> str:
        return "prana-deepseek-coder"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None,
                  **kwargs) -> ChatResult:
        role = {"system": "system", "human": "user", "ai": "assistant"}
        payload = [{"role": role.get(m.type, "user"), "content": str(m.content)}
                   for m in messages]
        out = self.client.generate(payload, max_new_tokens=self.max_new_tokens,
                                   adapter=self.use_adapter)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out["text"]))])


def build_chain(llm: BaseChatModel):
    """prompt | model | JSON parser, with one LangChain fallback: when the reply
    is prose instead of JSON (the base model does this on ~1 in 4 hunks), the
    same question is asked again with an explicit final "JSON only" turn."""
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    strict = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN + STRICT)])
    primary = prompt | llm | JsonOutputParser()
    retry = strict | llm | JsonOutputParser()
    return primary.with_fallbacks([retry], exceptions_to_handle=(OutputParserException,))


def review_security(chain, hunk: str, language: str, path: str) -> list[Finding]:
    """Validated findings for one hunk; [] on any parse or validation failure."""
    try:
        raw = chain.invoke({"hunk": hunk, "language": language, "path": path})
    except Exception as e:  # noqa: BLE001 - the security pass must never break a review
        log.warning("security pass failed for %s: %s", path, e)
        return []
    if isinstance(raw, list):                        # a bare list of findings
        raw = {"findings": raw}
    try:
        return SecurityReport.model_validate(raw).findings
    except ValidationError as e:
        log.warning("security pass output rejected for %s: %s", path, str(e)[:200])
        return []
