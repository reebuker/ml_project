import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from model_handler import ResnetClassifier

wts_path = "data/models/resnet18_weights.pth"

class ImageClassifierUI:
    def __init__(self, root, classifier: ResnetClassifier):
        self.root = root
        self.classifier = classifier

        self.root.title("Классификатор изображений")
        self.root.geometry("450x550")

        # Button
        self.btn_load = tk.Button(
            root, text = "Загрузить картинку", command=self.process_image
        )
        self.btn_load.pack(pady=20)

        # Виджет для картинки
        self.img_label = tk.Label(root)
        self.img_label.pack()

        self.result_label = tk.Label(
            root, text="Загрузите изображение..."
        )
        self.result_label.pack(pady=20)

    def process_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", ".jpg .jpeg .png .bmp")]
        )
        if not file_path:
            return

        try:
            predictions, visual_img = self.classifier.predict_and_visualize(file_path)

            # Отображаем раскрашенное изображение
            visual_img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(visual_img)
            self.img_label.configure(image=img_tk)
            self.img_label.image = img_tk

            display_text = "Результаты анализа:\n"
            for label, prob in predictions.items():
                display_text += f"* {label}: {prob * 100:.2f}%\n"

            self.result_label.configure(text=display_text)
        
        except Exception as e:
            self.result_label.configure(text=f"Ошибка анализа:\n{str(e)}")

if (__name__ == "__main__"):
    # Инициализация класса нейросети, может занять 1-2 секунды
    classifier_backend = ResnetClassifier(wts_path, 4)

    window = tk.Tk()
    app = ImageClassifierUI(window, classifier_backend)
    window.mainloop()