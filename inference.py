import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import time
import argparse

class LaneDetector:
    def __init__(self, model_path, device=None):
        """Initialize the model and load weights."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--> Using device: {self.device}")

        # Network architecture (must match training setup)
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )

        # Load model weights
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print("--> Model loaded successfully.")
        except FileNotFoundError:
            print(f"ERROR: File not found: {model_path}")
            exit()

    def preprocess(self, frame):
        """Resize, convert, normalize and convert frame to tensor."""
        img_resized = cv2.resize(frame, (512, 256))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_tensor = img_rgb.astype("float32") / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)

        return img_tensor.to(self.device)

    def predict(self, frame):
        """Run inference and return binary lane mask in original size."""
        original_h, original_w = frame.shape[:2]
        input_tensor = self.preprocess(frame)

        with torch.no_grad():
            logits = self.model(input_tensor)
            pred_mask = torch.sigmoid(logits).cpu().numpy()[0, 0]

        mask_resized = cv2.resize(pred_mask, (original_w, original_h))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        return binary_mask


def run_inference(video_source=0, model_path="lane_model_epoch_10.pth"):
    detector = LaneDetector(model_path=model_path)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("ERROR: Could not open video source.")
        return

    print("--> Starting inference... (press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        mask = detector.predict(frame)

        # Create a green overlay where the mask is active
        green_layer = np.zeros_like(frame)
        green_layer[:, :] = [0, 255, 0]
        green_layer = cv2.bitwise_and(green_layer, green_layer, mask=mask)

        output_frame = cv2.addWeighted(frame, 1.0, green_layer, 0.5, 0)

        fps = 1 / (time.time() - start_time)
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Lane Detection Real-Time", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video path or '0' for webcam")
    parser.add_argument("--model", default="lane_model_epoch_10.pth", help="Path to .pth file")

    args = parser.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    run_inference(video_source=src, model_path=args.model)
