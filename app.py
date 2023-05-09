from flask import Flask, request, send_file, render_template
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
)
import torch
from PIL import Image
from io import BytesIO
import requests
from flask_cors import CORS, cross_origin
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


app = Flask(__name__, template_folder="frontend", static_folder="frontend")
CORS(app, support_credentials=True)

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def cprint(msg: str, color: str = "blue", **kwargs) -> None:
    color_codes = {
        "blue": "\033[34m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "purple": "\033[35m",
        "cyan": "\033[36m",
    }
    
    if color not in color_codes:
        raise ValueError(f"Invalid info color: `{color}`")
    
    print(color_codes[color] + msg + "\033[0m", **kwargs)

model_name = "stabilityai/stablelm-tuned-alpha-7b"
#model_name = "stabilityai/stablelm-base-alpha-7b"
cprint(f"Using `{model_name}`", color="blue")

# Select "big model inference" parameters
torch_dtype = "float16" #@param ["float16", "bfloat16", "float"]
load_in_8bit = False #@param {type:"boolean"}
device_map = "auto"

cprint(f"Loading with: `{torch_dtype=}, {load_in_8bit=}, {device_map=}`")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=getattr(torch, torch_dtype),
    load_in_8bit=load_in_8bit,
    device_map=device_map,
    offload_folder="./offload",
)
model.half().cuda()

@app.route("/")
def index():
    return render_template("index.html")

@app.get("/health")
def health_check():
    return "Healthy", 200

@app.post("/txt2img")
def text_to_img():
    data = request.json
    #model_id = "andite/anything-v4.0" # 默认的，高品质、高细节的动漫风格
    #model_id = 'Envvi/Inkpunk-Diffusion' # 温克朋克风格，提示词 nvinkpunk
    #model_id = 'nousr/robo-diffusion-2-base' # 看起来很酷的机器人，提示词 nousr robot
    #model_id = 'prompthero/openjourney' # openjorney 风格,提示词 mdjrny-v4 style
    #model_id = 'dreamlike-art/dreamlike-photoreal-2.0' #写实，真实风格，提示词 photo
    model_id = "stabilityai/stable-diffusion-2"
    output = "output_txt2img.png"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    image = pipe(data["prompt"], guidance_scale=7.5, num_inference_steps=20,height=data["height"], width=data["width"]).images[0]

    image.save(output)
    return send_file(output), 200

@app.post("/img2img")
def img_to_img():
    data = request.json
    model_id = "runwayml/stable-diffusion-v1-5"
    output = "output_img2img.png"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    response = requests.get(data["url"])
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 512))
    images = pipe(
        prompt=data["prompt"], image=init_image, strength=0.75, guidance_scale=7.5
    ).images

    images[0].save(output)
    return send_file(output), 200

@app.post("/chat")
def chat():
    data = request.json
    
    user_prompt = data["user_content"]
    system_prompt = data["system_content"]

    prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"

    # Sampling args
    max_new_tokens = 128 #@param {type:"slider", min:32.0, max:3072.0, step:32}
    temperature = 0.7 #@param {type:"slider", min:0.0, max:1.25, step:0.05}
    top_k = 0 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
    top_p = 0.9 #@param {type:"slider", min:0.0, max:1.0, step:0.05}
    do_sample = True #@param {type:"boolean"}

    cprint(f"Sampling with: `{max_new_tokens=}, {temperature=}, {top_k=}, {top_p=}, {do_sample=}`")

    # Create `generate` inputs
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)

    # Generate
    tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )

    # Extract out only the completion tokens
    completion_tokens = tokens[0][inputs['input_ids'].size(1):]
    completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

    # Display
    print(user_prompt + " ", end="")
    cprint(completion, color="green")

    return completion, 200

app.run(host='0.0.0.0', port=5000)
