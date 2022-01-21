import digit_classifier
import os
os.environ["KCFG_KIVY_LOG_LEVEL"] = "warning"

from kivy.config import Config
Config.set("graphics", "resizable", False)

import kivy
kivy.require("2.0.0")

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stencilview import StencilView
from kivy.graphics import Line, Ellipse
from kivy.properties import ObjectProperty
from kivy.core.window import Window

Window.size = (480, 640)


# StencilView will not allow the user to draw out of the view
class PaintWidget(StencilView):
    def on_touch_down(self, touch):
        with self.canvas:
            # The ellipse here is used to create a starting point if you touch down but don't move
            size = 40
            Ellipse(pos=(touch.x - size / 2, touch.y - size / 2), size=(size, size))
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=size / 2)

    def on_touch_move(self, touch):
        touch.ud["line"].points += (touch.x, touch.y)


class RootWindow(GridLayout):
    paint_widget = ObjectProperty(None)
    prediction_label = ObjectProperty(None)

    def clear_canvas(self):
        self.paint_widget.canvas.clear()
        self.prediction_label.text = "Prediction:"

    def predict_digit(self):
        file_name = "predict_img.png"

        self.paint_widget.export_to_png(file_name)  # Save the image

        prediction = digit_classifier.predict_digit(folder=".", filename=file_name)  # Generate a prediction

        self.prediction_label.text = "Prediction: " + prediction  # Update the label to show the prediction

        os.remove(file_name)  # Remove the file


class DigitPredictorApp(App):

    def build(self):
        return RootWindow()


if __name__ == '__main__':
    DigitPredictorApp().run()
