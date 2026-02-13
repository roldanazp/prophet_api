import runpod
from tabpfn import TabPFNClassifier

def handler(job):
    try:
        job_input = job["input"]
        random_state = job_input["random_state"]
        x_context = job_input["x_context"]
        y_context = job_input["y_context"]
        x_target = job_input["x_target"]

        model = TabPFNClassifier(
            random_state=random_state,
            model_path="tabpfn-v2.5-classifier-v2.5_default.ckpt"
        )

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