"""
Gemini LLM Provider Implementation.
Uses the modern Google GenAI SDK (google-genai) with full async streaming,
tool calling, and error resilience.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional
from google import genai
from google.genai import errors, types

from app.core.config import settings
from app.ai.exceptions import (
    LLMAPIError,
    LLMConfigurationError,
    LLMInvalidResponseError,
    LLMTimeoutError,
)
from app.ai.llm.base import LLMProvider
from app.ai.prompt import strip_hidden_reasoning
from app.ai.schemas import ChatMessage, LLMResponse, ToolCall



class GeminiProvider(LLMProvider):
    """
    Production-ready Gemini LLM Provider implementing LLMProvider.
    Uses official google-genai SDK for Gemini 2.5 / 2.0 / 1.5 models.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client: Optional[genai.Client] = None
        if self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        return settings.GEMINI_MODEL

    def _get_client(self) -> genai.Client:
        """Retrieves or validates active GenAI client tied to current event loop."""
        if not self.api_key:
            raise LLMConfigurationError(
                "Gemini API Key is not configured. Please set GEMINI_API_KEY in backend/.env"
            )

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is None or getattr(self, "_loop", None) is not current_loop:
            self._client = genai.Client(api_key=self.api_key)
            self._loop = current_loop

        return self._client

    def _convert_messages(
        self, messages: List[ChatMessage]
    ) -> List[types.Content]:
        """
        Converts internal ChatMessage sequence to Gemini types.Content objects.
        Ensures valid role mapping ('user', 'assistant' -> 'model', 'tool' -> 'user', 'system' -> 'user')
        and merges consecutive turns with the same role into multiple parts.
        """
        contents: List[types.Content] = []
        for msg in messages:
            # Map role
            if msg.role in ("user", "human"):
                role = "user"
            elif msg.role in ("assistant", "model", "ai"):
                role = "model"
            elif msg.role in ("tool", "system"):
                # Non-user/model turns in message history mapped cleanly for Gemini
                role = "user"
            else:
                role = "user"

            # Merge consecutive messages with the same role
            if contents and contents[-1].role == role:
                contents[-1].parts.append(types.Part.from_text(text=msg.content))
            else:
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
        return contents

    async def generate(
        self,
        messages: List[ChatMessage],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generates a complete, non-streamed response from Gemini."""
        client = self._get_client()
        target_model = model or self.default_model
        timeout_val = timeout or settings.GEMINI_TIMEOUT_SECONDS

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature if temperature is not None else settings.GEMINI_TEMPERATURE,
            max_output_tokens=max_output_tokens or settings.GEMINI_MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = self._convert_messages(messages)

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_val,
            )

            # Extract text only, filtering out any hidden reasoning or thought parts
            text_parts = []
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts or []:
                    if getattr(part, "thought", False):
                        continue
                    if getattr(part, "text", None):
                        text_parts.append(part.text)

            raw_text = "".join(text_parts) if text_parts else (response.text or "")
            clean_text = strip_hidden_reasoning(raw_text)

            if not clean_text and not (response.function_calls if hasattr(response, "function_calls") else None):
                raise LLMInvalidResponseError(
                    "Gemini returned an empty response or output was blocked by safety filters."
                )

            finish_reason = None
            if response.candidates and response.candidates[0].finish_reason:
                finish_reason = str(response.candidates[0].finish_reason)

            prompt_tokens = None
            completion_tokens = None
            if response.usage_metadata:
                prompt_tokens = response.usage_metadata.prompt_token_count
                completion_tokens = response.usage_metadata.candidates_token_count

            return LLMResponse(
                content=clean_text.strip(),
                model=target_model,
                provider=self.provider_name,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Gemini request timed out after {timeout_val} seconds"
            )
        except errors.APIError as api_err:
            raise LLMAPIError(f"Gemini API Error: {str(api_err)}", status_code=502)
        except Exception as exc:
            if isinstance(exc, (LLMConfigurationError, LLMInvalidResponseError, LLMTimeoutError)):
                raise
            raise LLMAPIError(f"Gemini generation failed: {str(exc)}", status_code=500)

    async def stream(
        self,
        messages: List[ChatMessage],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streams response chunks from Gemini in real time.
        Yields partial text tokens as they arrive, skipping any hidden reasoning / thoughts.
        """
        client = self._get_client()
        target_model = model or self.default_model
        timeout_val = timeout or settings.GEMINI_TIMEOUT_SECONDS

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature if temperature is not None else settings.GEMINI_TEMPERATURE,
            max_output_tokens=max_output_tokens or settings.GEMINI_MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = self._convert_messages(messages)

        try:
            stream_response = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=target_model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_val,
            )

            async for chunk in stream_response:
                chunk_text = ""
                # Filter out thought parts from thinking models
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for cand in chunk.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if getattr(part, "thought", False):
                                    continue
                                if getattr(part, "text", None):
                                    chunk_text += part.text
                elif getattr(chunk, "text", None):
                    chunk_text = chunk.text

                if chunk_text:
                    cleaned_chunk = strip_hidden_reasoning(chunk_text)
                    if cleaned_chunk:
                        yield cleaned_chunk

        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Gemini stream initiation timed out after {timeout_val} seconds"
            )
        except errors.APIError as api_err:
            raise LLMAPIError(f"Gemini API Stream Error: {str(api_err)}", status_code=502)
        except Exception as exc:
            if isinstance(exc, (LLMConfigurationError, LLMTimeoutError)):
                raise
            raise LLMAPIError(f"Gemini streaming failed: {str(exc)}", status_code=500)

    async def generate_with_tools(
        self,
        messages: List[ChatMessage],
        tools: List[Any],
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generates a response with tool declarations."""
        client = self._get_client()
        target_model = model or self.default_model
        timeout_val = timeout or settings.GEMINI_TIMEOUT_SECONDS

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=temperature if temperature is not None else settings.GEMINI_TEMPERATURE,
            max_output_tokens=max_output_tokens or settings.GEMINI_MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = self._convert_messages(messages)

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_val,
            )

            tool_calls: Optional[List[ToolCall]] = None
            if response.function_calls:
                tool_calls = []
                for idx, call in enumerate(response.function_calls):
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{idx}_{call.name}",
                            name=call.name,
                            arguments=dict(call.args or {}),
                        )
                    )

            raw_candidate_content = (
                response.candidates[0].content if response.candidates else None
            )

            return LLMResponse(
                content=(response.text or "").strip(),
                model=target_model,
                provider=self.provider_name,
                tool_calls=tool_calls,
                raw_candidate_content=raw_candidate_content,
            )

        except asyncio.TimeoutError:
            raise LLMTimeoutError(f"Gemini request timed out after {timeout_val} seconds")
        except errors.APIError as api_err:
            raise LLMAPIError(f"Gemini API Error: {str(api_err)}", status_code=502)
        except Exception as exc:
            raise LLMAPIError(f"Gemini tool generation failed: {str(exc)}", status_code=500)

    async def stream_tool_response(
        self,
        messages: List[ChatMessage],
        raw_model_content: Any,
        tool_results: List[Dict[str, Any]],
        tools: Optional[List[Any]] = None,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Streams the final spoken response from Gemini after supplying tool execution results.
        Preserves model candidate content with thought signatures and passes tool output
        as FunctionResponse parts for natural synthesis.
        """
        client = self._get_client()
        target_model = model or self.default_model
        timeout_val = timeout or settings.GEMINI_TIMEOUT_SECONDS

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=temperature if temperature is not None else settings.GEMINI_TEMPERATURE,
            max_output_tokens=max_output_tokens or settings.GEMINI_MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        contents = self._convert_messages(messages)
        if raw_model_content:
            contents.append(raw_model_content)

        # Append function responses
        response_parts = []
        for tr in tool_results:
            response_parts.append(
                types.Part.from_function_response(
                    name=tr["name"],
                    response=tr["response"],
                )
            )

        if response_parts:
            contents.append(
                types.Content(
                    role="user",
                    parts=response_parts,
                )
            )

        try:
            stream_response = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=target_model,
                    contents=contents,
                    config=config,
                ),
                timeout=timeout_val,
            )

            async for chunk in stream_response:
                chunk_text = ""
                if hasattr(chunk, "candidates") and chunk.candidates:
                    for cand in chunk.candidates:
                        if cand.content and cand.content.parts:
                            for part in cand.content.parts:
                                if getattr(part, "thought", False):
                                    continue
                                if getattr(part, "text", None):
                                    chunk_text += part.text
                elif getattr(chunk, "text", None):
                    chunk_text = chunk.text

                if chunk_text:
                    cleaned_chunk = strip_hidden_reasoning(chunk_text)
                    if cleaned_chunk:
                        yield cleaned_chunk

        except asyncio.TimeoutError:
            raise LLMTimeoutError(
                f"Gemini stream initiation timed out after {timeout_val} seconds"
            )
        except errors.APIError as api_err:
            raise LLMAPIError(f"Gemini API Stream Error: {str(api_err)}", status_code=502)
        except Exception as exc:
            if isinstance(exc, (LLMConfigurationError, LLMTimeoutError)):
                raise
            raise LLMAPIError(f"Gemini streaming failed: {str(exc)}", status_code=500)
