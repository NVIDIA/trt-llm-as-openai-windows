# 🚀 TensorRT-LLM as OpenAI API on Windows 🤖

Drop-in replacement REST API compatible with OpenAI API spec using TensorRT-LLM as the inference backend.

Setup a local Llama 2 or Code Llama web server using TRT-LLM for compatibility with the OpenAI Chat and legacy Completions API. This enables accelerated inference on Windows natively, while retaining compatibility with the wide array of projects built using the OpenAI API.

Follow this README to setup your own web server for Llama 2 and Code Llama.

## Getting Started

Ensure you have the pre-requisites in place:

1. Install [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/) for Windows using the instructions [here](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.6.1/windows#quick-start).

2. Ensure you have access to the Llama 2 repository on Huggingface
   * [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
   * [CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)

3. In this repo, we provide instructions to set up an OpenAI API compatible server with either the LLaMa 2 13B or CodeLlama 13B model, both optimized using AWQ 4-bit quantization. To begin, it's necessary to compile a TensorRT Engine tailored to your specific GPU. For users with the GeForce RTX 4090, combined with TensorRT-LLM release v0.6.1, the pre-compiled TRT Engines are available for download through the provided links below. If utilizing a different NVIDIA GPU or another version of TensorRT, refer to the given instructions for constructing your TRT Engine [instructions](#building-trt-engine).

   #### Prebuilt TRT Engine

   | # |  Model                | RTX GPU   | TensorRT Engine <br/>(TRT-LLM 0.6.1)|  
   |---|----------------------------------|---------|---------------------------------------|
   | 1 | Llama2-13b-chat AWQ int4         | RTX 4090 | [Download](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2) |
   | 2 | CodeLlama-13b-instruct AWQ int4 | RTX 4090 |  TODO: <br/> ```\\nvsw-dump\users\aaka\winAi\code_llama_13b_instruct_ammo_awq_engine_16k\eng```       |


<h3 id="setup"> Setup Steps </h3>

1. Clone this repository: 
   ```
   https://github.com/NVIDIA/trt-llm-as-openai-windows
   cd trt-llm-as-openai-windows
   ```
2. Download the tokenizer and config.json from HuggingFace and place them in the model/ directory.
   - [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main).
   - [CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/tree/main)
3. Place the TensorRT engine & config.json (from NGC) for the Llama/CodeLlama model in the 'model/engine' directory
   - For GeForce RTX 4090 users: Download the pre-built TRT engine and config.json for [Llama2-13B](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2) or [CodeLlama-13B](update NGC link) and place it in the model/ directory.
   - For other NVIDIA GPU users: Build the TRT engine by following the instructions provided [here](#building-trt-engine).
4. Install the necessary libraries: 
   ```
   pip install -r requirements.txt
   ```
5. Launch the application using the following command:

   - LlaMa-2-13B-chat model
   ```
   python app.py --trt_engine_path <TRT Engine folder> --trt_engine_name <TRT Engine file>.engine --tokenizer_dir_path <tokernizer folder> --port <optional port>
   ```
   
   - CodeLlaMa-13B-instruct model needs below additional commands
   ```
   --max_output_tokens 4096 --max_input_tokens 2048 --no_system_prompt True
   ```
   In our case, that will be (for Llama-2):
   ```
   python app.py --trt_engine_path model/ --trt_engine_name llama_float16_tp1_rank0.engine --tokenizer_dir_path model/ --port 8081
   ```

### Test the API

1. Install the 'openai' client library in your Python environment.
   ```
   pip install openai
   ```
  
2. Run the following code inside your Python env.
   
3. Set a random API key and the base URL.
<pre>
openai.api_key="ABC"  
openai.api_base="http://127.0.0.1:8081"
response = openai.ChatCompletion.create(
  model = "Llama2",
  messages = [{"role": "user", "content": "Hello!"}]
  )
print(response)
</pre>
   

## Detailed Command References 
```
python app.py --trt_engine_path <TRT Engine folder> --trt_engine_name <TRT Engine file>.engine --tokenizer_dir_path <tokernizer folder> --port <port>
```

Arguments

| Name                    | Details                     |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------|
| --trt_engine_path <>    | Directory of TensorRT engine ([Prebuilt TRT Engine](#prebuilt-trt-engine) or locally built TRT engine [instructions](#building-trt-engine))                                                                                                                        |
| --trt_engine_name <>    | Engine file name (e.g. llama_float16_tp1_rank0.engine)                                                                                                      |
| --tokenizer_dir_path <> | HF downloaded model files for tokenizer & config.json e.g. [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) or [CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/tree/main) |
| --port <>               | OpenAI compatible server hosted on localhost and 8081 port as default. Optionally, allows to specify a different port.  |
| --max_output_tokens     | Optional override to maximum output token sizes otherwise it defaults to 2048 |
| --max_input_tokens      | Optional override to maximum input token sizes otherwise it defaults to 2048 |
| --no_system_prompt      | App uses default system prompt and optionally supported to avoid it. | 


<h3 id="supported-apis">Supported APIs</h3>

* /completions
* /chat/completions
* /v1/completions
* /v1/chat/completions

<h2 id="use-cases">Examples</h3>
<h3> <a href="https://continue.dev">Continue.dev</a> Visual Studio Code Extension with CodeLlama-13B </h3>

1. Run this app with CodeLlama-13B-instruct AWQ int4 model as described above.
2. Install Continue.dev from [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
3. Configure to use OpenAI API compatible local inference from UI
   1. Open Continue.dev plugin from Visual Studio Code left panel
   2. Click "+" to add new model
   3. Select "Other OpenAI-compatible API"
   4. Expand "Advanced (optional)"
      1. server_url: update to local host url like ```http://localhost:8081/v1```
      2. update context_length: ```16384```
   5. Select CodeLlama 13b instruct option 
   6. Press "Configure Model in config.py"
4. Alternatively config.py can be modified directly to include below
   1. Open ```c:\users\<user>\.continue\config.py``` in any editor
   2. Add below model config
      ```
      from continuedev.libs.llm.ggml import GGML
      ...
      config = ContinueConfig(
       allow_anonymous_telemetry=False,
       models=Models(
               default=GGML(
                  title="CodeLlama-13b-Instruct",
                  model="codellama:13b-instruct",
                  context_length=16384,
                  server_url="http://localhost:8081"
               ),
      ...
      ```

<h3 id="building-trt-engine">Building TRT Engine</h3>

For RTX 4090 (TensorRT-LLM v0.6.1), a prebuilt TRT engine is provided. For other RTX GPUs or TensorRT versions, follow these steps to build your TRT engine:

Download models and quantized weights
  * Llama-2-13b-chat AWQ int4 
    * Download Llama-2-13b-chat model from [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
    * Download Llama-2-13b-chat AWQ int4 checkpoints from [here](https://catalog.ngc.nvidia.com/orgs/nvidia/models/llama2-13b/files?version=1.2)
  * CodeLlama-13B-instruct AWQ int
    * Download CodeLLaMa-2 13B model from [CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)
    * Download CodeLLaMa 2 13B AWQ int4 checkpoints from [\\nvsw-dump\users\aaka\winAi\code_llama_13b_instruct_ammo_awq_engine_16k]


Clone the [TensorRT LLM](https://github.com/NVIDIA/TensorRT-LLM/) repository:
```
git clone https://github.com/NVIDIA/TensorRT-LLM.git
```

Navigate to the examples\llama directory and run the following script:
```
python build.py --model_dir <path to llama13_chat model> --quant_ckpt_path <path to model.pt> --dtype float16 --use_gpt_attention_plugin float16 --use_gemm_plugin float16 --use_weight_only --weight_only_precision int4_awq --per_group --enable_context_fmha --max_batch_size 1 --max_input_len 3500 --max_output_len 1024 --output_dir <TRT engine folder>
```


This project requires additional third-party open source software projects as specified in the documentation. Review the license terms of these open source projects before use.
