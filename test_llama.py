import datasets
from datasets import load_dataset
from openai import OpenAI
import json
import regex as re

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
#   api_key="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"
#   api_key="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",
)

def call_llama(question):
    completion = client.chat.completions.create(
    extra_headers={
    },
    extra_body={},
    model="meta-llama/llama-3.3-8b-instruct:free",
    messages=[
        {
        "role": "user",
        "content": question
        }
    ]
    )
    return completion.choices[0].message.content

def test_llama():
    question = "If the point \\(P(\\sin\\theta\\cos\\theta, 2\\cos\\theta)\\) is located in the third quadrant, then angle \\(\\theta\\) belongs to the quadrant number ___. Given the information: (1) if $ab < 0$ and $b < 0$, then $a > 0$, and (2) quadrant I: both sin and cos are positive, quadrant II: sin is positive, cos is negative, quadrant III: both sin and cos are negative, quadrant IV: sin is negative, cos is positive."
    # question = "Given \\sin A = 4(1 - \\cos A), then $\\sin A=$ _____."
    ans = call_llama(question)
    print(ans)
    # print(json.dumps({"ans": ans}))

def read_results(path):
    results = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                l = json.loads(line)
                q = l["problem"]
                results[q] = l
    return results

def write_results(path, results):
    with open(path, mode="w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    

def run_solve_math(threshold=0.1, res_path="results.jsonl", limit=100):
    results = read_results(res_path)
    print("Loaded {} previous results".format(len(results)))
    ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified")
    ans_pattern = re.compile(r'\\boxed\s*\{((?:[^{}]|\{(?1)\})*)\}')
    cnt, skip = 0, 0
    for row in ds["train"]:
        question = row["problem"]
        if question in results:
            skip += 1
            print("Skip {} questions".format(skip))
        elif row["llama8b_solve_rate"] < threshold:
            try:
                response = call_llama(question)
                answer = ans_pattern.findall(response)
                row["response"] = response
                row["llama_answer"] = answer[0] if len(answer) > 0 else ""
                results[question] = row
            except Exception as e:
                print(e)
            cnt += 1
            print("Solved {} cases".format(cnt))
            if cnt == limit:
                break
    res = []
    for k, v in results.items():
        res.append(v)
    write_results(res_path, res)

def check_db():
    ds = load_dataset("ddrg/named_math_formulas")
    cnt = 0
    key = "quadrant"
    for row in ds["train"]:        
        cnt += 1
        if cnt % 50000 == 0:
            print("cnt =", cnt)
        # if cnt > 2000000:
        #     break
        if key in row["name"] or key in row["formula"]:
            print(row)

if __name__ == "__main__":
    # test_llama()
    # check_db()
    test_llama()
    # print("## Step 1: Convert the length of the sides from li to meters\nFirst, we need to convert the length of the sides of the triangular sand field from li to meters, knowing that 1 li is equivalent to 300 steps. However, the problem does not specify the length of one step in meters, so we will assume that the conversion is directly from li to meters without needing the step length. This assumption is based on the provided information and the context of the problem, which implies a direct conversion for the calculation of the circumcircle's radius.\n\n## Step 2: Calculate the semi-perimeter of the triangle\nThe formula to calculate the semi-perimeter (s) of a triangle when all sides are known is s = (a + b + c) / 2, where a, b, and c are the lengths of the sides. In this case, a = 13 li, b = 14 li, and c = 15 li. We will convert these lengths to meters after understanding that the conversion factor is not explicitly needed for the semi-perimeter calculation.\n\n## Step 3: Apply Heron's formula to find the area of the triangle\nHeron's formula states that the area (A) of a triangle can be found using the formula A = sqrt(s(s-a)(s-b)(s-c)), where s is the semi-perimeter calculated in the previous step.\n\n## Step 4: Use the relationship between the area and the circumradius\nThe area (A) of a triangle is also related to its circumradius (R) by the formula A = (abc) / (4R), where a, b, and c are the sides of the triangle. We can rearrange this formula to find R = (abc) / (4A).\n\n## Step 5: Calculate the semi-perimeter in meters\nSince we are working with li and the conversion to meters is not directly applicable without a conversion factor for li to meters, we recognize that the calculation of the semi-perimeter and the area must be done with the understanding that li is a unit of measurement that needs conversion. However, the problem's phrasing suggests that the conversion should directly apply to the final calculation of the radius, implying that we should focus on the relationship between the sides and the circumradius rather than the conversion of li to meters for the intermediate steps.\n\n## Step 6: Calculate the area using Heron's formula and find the circumradius\nGiven the sides a = 13, b = 14, and c = 15, we can calculate the semi-perimeter and then the area using Heron's formula. However, we recognize that the direct conversion to meters is implied for the final answer but not explicitly needed for the intermediate calculations based on the information provided.\n\n## Step 7: Final calculation of the circumradius\nWe will calculate the semi-perimeter and then apply Heron's formula to find the area. With the area known, we can calculate the circumradius using the formula R = (abc) / (4A).\n\n## Step 8: Execute the semi-perimeter calculation\ns = (13 + 14 + 15) / 2 = 42 / 2 = 21.\n\n## Step 9: Execute Heron's formula\nA = sqrt(21(21-13)(21-14)(21-15)) = sqrt(21 * 8 * 7 * 6) = sqrt(7056) = 84.\n\n## Step 10: Calculate the circumradius\nR = (13 * 14 * 15) / (4 * 84) = 2730 / 336 = 65 / 8.\n\nThe final answer is: $\\boxed{8.125}$")
    # run_solve_math()
    print(json.dumps({"a": ""}))
