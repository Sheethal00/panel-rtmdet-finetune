from harness.backends import RTMDetOnnx
import cv2

det = RTMDetOnnx(
    "exports/rtmdet_tiny_panel.onnx",
    class_map={0: "MCB", 1: "RCBO", 2: "RCCB"},
)
img = cv2.imread("panel_images/11_1.jpeg")  # known ground truth: 7 devices
results = det.detect(img)
print(f"{len(results)} devices found")
for d in results:
    print(f"  {d.label:6s} score={d.score:.3f}  box={d.box}")