from pydantic import BaseModel
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    input_guardrail,
    output_guardrail,
    RunContextWrapper,
    set_tracing_disabled,
)

# ----------------------------------
# 🔐 1. Environment & API Setup
# ----------------------------------
load_dotenv()
set_tracing_disabled(True)

API_KEY = os.getenv("API_KEY")

external_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# ----------------------------------
# 🛡️ 2. Input Guardrail Setup
# ----------------------------------
class MathsHomeworkOutput(BaseModel):
    Is_math_work: bool
    resoning: str

Guardrail_Agent = Agent(
    name="Input Guardrail Agent",
    instructions=(
        "Determine if the user is directly asking you to *do* their math homework or solve an assignment for them. "
        "If the user is simply asking for an explanation or how to approach a problem, set Is_math_work=False. "
        "Only set Is_math_work=True if they are trying to copy a math problem for you to solve entirely."
    ),
    model=model,
    output_type=MathsHomeworkOutput
)


@input_guardrail
async def maths_guardrail(ctx: RunContextWrapper, agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(Guardrail_Agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.Is_math_work
    )

# ----------------------------------
# ✅ 3. Output Guardrail Setup
# ----------------------------------
class OutputCheck(BaseModel):
    valid_solution: bool
    explanation: str

Output_guardrail_agent = Agent(
    name="Output Guardrail Agent",
    instructions="Check if the response includes a valid numeric solution and steps. Return valid_solution=True if it is valid.",
    model=model,
    output_type=OutputCheck
)

@output_guardrail
async def valid_output_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    agent_output: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(Output_guardrail_agent, agent_output, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.valid_solution
    )




# ----------------------------------
# 🧠 4. Main Agent
# ----------------------------------
agent = Agent(
    name="Math Solver Agent",
    instructions="Solve the math question step-by-step and include a numeric answer.",
    model=model,
    input_guardrails=[maths_guardrail],
    output_guardrails=[valid_output_guardrail]
)

# ----------------------------------
# 🚀 5. Run Agent
# ----------------------------------
async def main():
    try:
        result = await Runner.run(
            agent,
            input="how can you explain 2x + 3 = 11?"
        )
        print("✅ Agent Response:")
        print(result.final_output)

    except InputGuardrailTripwireTriggered:
        print("🚫 Input Guardrail Blocked: This looks like math homework.")

    except OutputGuardrailTripwireTriggered:
        print("⚠️ Output Guardrail Blocked: Invalid or unclear math solution.")

# Entry point
if __name__ == "__main__":
    asyncio.run(main())



