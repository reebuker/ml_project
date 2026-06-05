import tkinter as tk
from PIL import ImageTk
from tkinter import filedialog
from model_handler import ResnetClassifier

wts_path = "resnet18_weights.pth"

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

        self.result_label = tk.Text(root, height=6, width=40, bd=0, bg=root.cget("bg"), font=("TkDefaultFont", 10))
        self.result_label.pack(pady=20)
        self.result_label.tag_configure("center", justify="center")
        self.result_label.insert("1.0", "Загрузите изображение...")
        self.result_label.tag_add("center", "1.0", "end")

        # Предупреждение (дисклеймер) внизу окна
        self.lbl_warning = tk.Label(
            root, 
            text="⚠️ Внимание: модель является учебной. Результаты классификации могут быть неточными.\nДля принятия важных решений проконсультируйтесь со специалистом.",
            font=("TkDefaultFont", 10, "italic"),
            fg="#666666",  # Серый цвет текста, чтобы не отвлекал
            justify="center"
        )
        # side="bottom" закрепляет элемент в самом низу, fill="x" растягивает по ширине
        self.lbl_warning.pack(side="bottom", pady=15, fill="x")


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

            # Разрешаем редактирование, очищаем поле и пишем заголовок
            self.result_label.config(state="normal")
            self.result_label.delete("1.0", tk.END)
            self.result_label.insert("end", "Результаты анализа:\n")

            # Находим самую высокую вероятность среди результатов
            max_prob = max(predictions.values()) if predictions else 0

            for label, prob in predictions.items():
                line_text = f"* {label}: {prob * 100:.2f}%\n"

                # Фиксируем координаты начала и конца строки
                start_idx = self.result_label.index("end-1c")
                self.result_label.insert("end", line_text)
                end_idx = self.result_label.index("end-1c")

                # Если это топ-1 предсказание, применяем стиль "bold"
                if prob == max_prob:
                    self.result_label.tag_add("bold", start_idx, end_idx)

            # Применяем выравнивание по правому краю ко всему тексту и блокируем поле
            self.result_label.tag_add("left", "1.0", "end")
            self.result_label.config(state="disabled")
        
        except Exception as e:
            self.result_label.config(state="normal")
            self.result_label.delete("1.0", tk.END)
            self.result_label.insert("1.0", f"Ошибка анализа:\n{str(e)}")
            self.result_label.tag_add("left", "1.0", "end")
            self.result_label.config(state="disabled")


if (__name__ == "__main__"):
    # Инициализация класса нейросети, может занять 1-2 секунды
    classifier_backend = ResnetClassifier(wts_path, 4)

    window = tk.Tk()
    app = ImageClassifierUI(window, classifier_backend)
    window.mainloop()