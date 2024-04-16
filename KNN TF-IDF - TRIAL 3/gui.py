import tkinter
from tkinter import ttk
import train

class TFIDF_KNN_UI:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.resizable(False, False)
        self.root.geometry("480x650")
        self.root.title("KNN TF-IDF")
        self.light_font_header = ("Segoe UI Light", 32)
        self.light_font_subheader1 = ("Segoe UI Light", 24)
        self.light_font_subheader2 = ("Segoe UI Semilight", 14)
        self.light_font_subheader3 = ("Segoe UI Semilight", 12)
        self.home()
        self.root.mainloop()


    def home(self):
        tkinter.Label(master=self.root, text="KNN TF-IDF", font=(self.light_font_header)).pack(pady=(24, 0))
        tkinter.Label(master=self.root, text="SMKN 4 MALANG", font=(self.light_font_subheader2)).pack()

        textbox = tkinter.Text(master=self.root, selectborderwidth=0, font=self.light_font_subheader3,border=None, highlightthickness=0, height=12, width=40, relief=tkinter.FLAT)
        textbox.pack(pady=(20, 8))
        
        self.result_textbox = tkinter.Text(master=self.root, selectborderwidth=0, font=self.light_font_subheader3,border=None, highlightthickness=0, height=1, width=40, relief=tkinter.FLAT)
        self.result_textbox.insert("1.0", " Sentiment :")
        self.result_textbox.configure(state="disabled")
        self.result_textbox.pack()

        button_predict = tkinter.Button(master=self.root, font=self.light_font_subheader2,text="Predict", relief=tkinter.FLAT, background="white",command=lambda: self.predict(str(textbox.get("1.0", "end-1c"))), width=31, height=1)
        button_predict.pack(pady=(20, 0))
        
        button_statistic = tkinter.Button(master=self.root, text="Statistic", font=self.light_font_subheader2,relief=tkinter.FLAT, background="white",command=lambda: self.predict(str(textbox.get("1.0", "end-1c"))), width=31, height=1)
        button_statistic.pack(pady=(10, 0))
        #progressbar = ttk.Progressbar(master=self.root, orient="horizontal", mode="determinate", length=200)
        #progressbar.pack()
        #progressbar["value"] = 100

    def statistic(self):
        pass

    def predict(self, text):
        text = f" Sentiment : {train.predict(text)}"

        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", tkinter.END)
        self.result_textbox.insert("1.0", text)
        self.result_textbox.configure(state="disabled")



ui = TFIDF_KNN_UI()
