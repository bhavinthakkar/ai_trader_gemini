import json
import ollama

raw_text = """
Apple's Q3 2026 numbers came in ahead of estimates. Earnings per share were $2.02 against a $1.89 consensus. Revenue was $109.4 billion versus the $108.8 billion estimate. 
iPhone revenue hit $54.2 billion, above the $53.5 billion projection.
"""

# Stage 1: FinGPT / Text Specialist
extraction_prompt = f"""
Analyze the following financial text. Extract key metrics and return ONLY a JSON object:
Text: {raw_text}
"""

stage1_response = ollama.chat(
    model="qwen2.5:14b",
    messages=[{"role": "user", "content": extraction_prompt}],
    format="json",
)
extracted_data = json.loads(stage1_response["message"]["content"])

# Stage 2: DeepSeek-R1 Reasoning Engine
reasoning_prompt = f"""
Act as a senior equity analyst. Analyze the structured financial data provided below:

Data: {json.dumps(extracted_data)}

Please perform the following calculations and multi-step reasoning:
1. Compare the top-line revenue growth against operating margin contraction. Calculate the net operational leverage efficiency.
2. Evaluate the risk of the Q4 EBITDA guidance drop (-5%) against the raised full-year revenue guidance. Does this signal margin compression or pricing pressure?
3. Calculate the implied gross profit and assess whether Free Cash Flow (-8%) deterioration threatens dividend/buyback capacity.
4. Provide a final Recommendation Score (1-10) with step-by-step risk justification.
"""


stage2_response = ollama.chat(
    model="deepseek-r1:14b",
    messages=[{"role": "user", "content": reasoning_prompt}],
)

print("### Final Financial Assessment ###\n")
print(stage2_response["message"]["content"])