import tkinter
import train
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class TFIDF_KNN_UI:
    def __init__(self):
        self.light_font_header = ("Segoe UI Light", 32)
        self.light_font_subheader1 = ("Segoe UI Light", 20)
        self.light_font_subheader2 = ("Segoe UI Semilight", 14)
        self.light_font_subheader3 = ("Segoe UI Semilight", 12)
        self.main_color = "#516c8d"
        self.secondary_color = "#28385e"
        self.background_color = "#304163"
        self.font_color = "white"
        
        self.root = tkinter.Tk()
        self.root.resizable(False, False)
        self.root.geometry("480x650")
        self.root.title("KNN TF-IDF")
        self.root.configure(background=self.background_color)
        self.home()
        self.root.mainloop()

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def home(self):
        self.clear()
        tkinter.Label(master=self.root, text="Prediksi Sentiment Game Dengan Algoritma TF-IDF", font=(self.light_font_header), background=self.background_color, foreground=self.font_color).pack(pady=(24, 0))
        tkinter.Label(master=self.root, text="SMKN 4 MALANG", font=(self.light_font_subheader1), background=self.background_color, foreground=self.font_color).pack()

        textbox = tkinter.Text(master=self.root, selectborderwidth=0, font=self.light_font_subheader3,border=None, highlightthickness=0, height=12, width=40, relief=tkinter.FLAT, background=self.main_color, foreground=self.font_color)
        textbox.pack(pady=(20, 8))
        
        self.result_textbox = tkinter.Text(master=self.root, selectborderwidth=0, font=self.light_font_subheader3,border=None, highlightthickness=0, height=0,width=40, relief=tkinter.FLAT, background=self.main_color, foreground=self.font_color)
        self.result_textbox.insert("1.0", " Sentiment :")
        self.result_textbox.configure(state="disabled")
        self.result_textbox.pack()

        button_predict = tkinter.Button(master=self.root, font=self.light_font_subheader2,text="Predict", relief=tkinter.FLAT, foreground=self.font_color, background=self.main_color,command=lambda: self.predict(str(textbox.get("1.0", "end-1c"))), width=31, height=1)
        button_predict.pack(pady=(20, 0))
        
        button_statistic = tkinter.Button(master=self.root, text="Statistic", font=self.light_font_subheader2,relief=tkinter.FLAT, foreground=self.font_color,background=self.main_color,command=self.statistic, width=31, height=1)
        button_statistic.pack(pady=(10, 0))

    def statistic(self):
        self.clear()
        tkinter.Label(master=self.root, text=f"Statistic", font=(self.light_font_header), background=self.background_color, foreground=self.font_color).pack(pady=40)
        tkinter.Label(master=self.root, text=f"Accuracy : {train.accuracy} %", font=(self.light_font_subheader1), background=self.background_color, foreground=self.font_color).pack()
        tkinter.Label(master=self.root, text=f"Train Data : {len(train.x_train)}", font=(self.light_font_subheader1), background=self.background_color, foreground=self.font_color).pack()
        tkinter.Label(master=self.root, text=f"Test Data : {len(train.x_test)}", font=(self.light_font_subheader1), background=self.background_color, foreground=self.font_color).pack()
        tkinter.Label(master=self.root, text=f"K Neighbors : {train.knn.n_neighbor}", font=(self.light_font_subheader1), background=self.background_color, foreground=self.font_color).pack()
        
        #plt.figure(figsize=(10, 5))
        #plt.imshow(train.wordcloud(), interpolation=bilinear)
        #plt.axis('off')

        #canvas = FigureCanvasTkAgg()
        
        button_predict = tkinter.Button(master=self.root, font=self.light_font_subheader2,text="Back", relief=tkinter.FLAT, foreground=self.font_color, background=self.main_color,command=self.home, width=31, height=1)
        button_predict.pack(pady=(20, 0))

    def predict(self, text):
        text = f" Sentiment : {train.predict(text)}"

        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", tkinter.END)
        self.result_textbox.insert("1.0", text)
        self.result_textbox.configure(state="disabled")



ui = TFIDF_KNN_UI()
