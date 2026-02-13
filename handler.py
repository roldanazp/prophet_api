import os

import runpod
from tabpfn import TabPFNClassifier

MODEL_ID = os.environ.get("MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Force offline mode to use only cached models
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def handler(job):
    try:
        job_input = job["input"]
        random_state = job_input["random_state"]
        x_context = job_input["x_context"]
        y_context = job_input["y_context"]
        x_target = job_input["x_target"]

        model = TabPFNClassifier(random_state=random_state)
        model.fit(x_context, y_context)
        predictions = model.predict(x_target)

        return {
            "status": "success",
            "predictions": predictions
        }

    except Exception as e:
        print(f"[Handler] Error during generation: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})