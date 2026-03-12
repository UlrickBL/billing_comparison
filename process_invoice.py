import torch
from transformers import (
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
    AutoTokenizer,
    AutoModelForCausalLM
)
from pdf2image import convert_from_path
from PIL import Image
import json


device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

dtype = torch.bfloat16


ocr_model = LightOnOcrForConditionalGeneration.from_pretrained(
    "lightonai/LightOnOCR-2-1B",
    torch_dtype=dtype
).to(device)

ocr_processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")


def ocr_image(image):

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text from this document."}
            ],
        }
    ]

    inputs = ocr_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    output_ids = ocr_model.generate(**inputs, max_new_tokens=2048)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]

    text = ocr_processor.decode(generated_ids, skip_special_tokens=True)

    return text


def ocr_pdf(pdf_path):

    images = convert_from_path(pdf_path)

    full_text = ""

    for img in images:
        text = ocr_image(img)
        full_text += text + "\n"

    return full_text


model_name = "Qwen/Qwen3.5-0.8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def extract_invoice(text):

    prompt = f"""
Extract invoice information.

Return ONLY JSON with this schema:

{{
  "invoice_number": "",
  "invoice_date": "",
  "supplier": "",
  "customer": "",
  "total_ht": "",
  "total_ttc": "",
  "products": [
    {{
      "reference": "",
      "description": "",
      "quantity": "",
      "unit_price": "",
      "total_price": ""
    }}
  ]
}}

Invoice text:
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=600,
        temperature=0
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    start = response.find("{")
    end = response.rfind("}") + 1

    print(response)
    return json.loads(response[start:end])