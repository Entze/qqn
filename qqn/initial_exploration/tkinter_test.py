from tkinter import *
from tkinter import ttk
from typing import Dict, Optional, Any


class CanvasElem:

    def in_convex_hull(self, point_x: int, point_y: int):
        return True


class Rectangle(CanvasElem):
    def __init__(self, top_left_x: int, top_left_y: int, width: int, height: int):
        self.top_left_x: int = top_left_x
        self.top_left_y: int = top_left_y
        self.width: int = width
        self.height: int = height

    def in_convex_hull(self, point_x: int, point_y: int):
        return self.top_left_x <= point_x <= self.top_left_x + self.width and \
               self.top_left_y <= point_y <= self.top_left_y + self.height


modes: Dict[str, Optional[Dict[str, Any]]] = {
    "View": None,
    "Move": {"Node": int},
    "InsertNode": None,
    "InsertEdge": None,
    "ConnectEdge": {"Node": int},
    "Debounce": None
}


class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.elems: Dict[int, CanvasElem] = {}
        self.lastx: int = 0
        self.lasty: int = 0
        self.sel_elem: Optional[int] = None
        self.state: str = "View"
        self.bind("<Button-1>", self.select_elem_at_point)
        self.bind("<B1-ButtonRelease>", self.clear_selection)
        self.bind("<KeyPress>", self.pressed_key)
        self.bind("<Button-1>", lambda event: self.focus_set(), add=True)
        self.bind("<Button-2>", lambda event: self.focus_set(), add=True)

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def select_elem_at_point(self, event):
        print(event)
        self.sel_elem = None
        for id, elem in self.elems.items():
            if elem.in_convex_hull(event.x, event.y):
                self.sel_elem = int
                break
        if self.sel_elem is not None:
            print(f"Selected elem with id {self.sel_elem}")

    def clear_selection(self, event):
        if self.sel_elem is not None:
            print(f"Clear selection of elem with id {self.sel_elem}")
        self.sel_elem = None

    def pressed_key(self, event):
        print(event)
        if event.keysym == "i":
            return self.pressed_i(event)

    def pressed_i(self, event):
        assert self.state in modes
        new_state = None
        if self.state == "View":
            new_state = "InsertNode"
        elif self.state == "InsertNode":
            new_state = "View"
        assert new_state in modes
        print(f"Change from {self.state}-State to {new_state}-State")
        self.state = new_state


def main():
    root = Tk()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    sketch = Sketchpad(root)
    sketch.grid(column=0, row=0, sticky=(N, W, E, S))

    root.mainloop()


if __name__ == "__main__":
    main()
