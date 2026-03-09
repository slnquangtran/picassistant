import json
from transformers import pipeline
from utils import load_image, pil_to_base64

class DepthEstimationAgent:
    def __init__(self, model_name="Intel/dpt-large"):
        """
        Initializes the depth estimation pipeline.
        """
        self.depth_estimator = pipeline("depth-estimation", model=model_name)

    def run(self, image_input):
        """
        Runs depth estimation on the input image.
        Returns a base64-encoded depth map.
        """
        image = load_image(image_input)
        result = self.depth_estimator(image)
        
        # The result from pipeline("depth-estimation") usually contains:
        # 'predicted_depth': torch.Tensor
        # 'depth': PIL Image (normalized)
        
        depth_image = result['depth']
        base64_depth_map = pil_to_base64(depth_image)

        return json.dumps({"depth_map": base64_depth_map})

if __name__ == "__main__":
    # Example usage (commented out)
    # agent = DepthEstimationAgent()
    # print(agent.run("path/to/image.jpg"))
    pass
