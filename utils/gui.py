import os
import sys
import json
import time
import importlib.util
import traceback
import argparse
from enum import Enum
from inspect import isclass

from PyQt5 import QtWidgets, QtCore

from discopygal.solvers import Scene, SceneDrawer, PathCollection, PathPoint
from discopygal.solvers.Solver import Solver
from discopygal.bindings import Point_2
from discopygal.gui.logger import Writer, Logger
from discopygal.gui.Worker import Worker
from discopygal.gui.gui import GUI
from discopygal.geometry_utils.display_arrangement import display_arrangement
from discopygal.solvers.verify_paths import verify_paths

try:
    from discopygal.solvers.rrt import RRT, dRRT, BiRRT
    from discopygal.solvers.prm import PRM, BasicRodPRM
    from discopygal.solvers.exact import ExactSingleDisc, ExactSinglePoly
except ImportError:
    pass


from discopygal_tools.solver_viewer.solver_viewer_gui import Ui_MainWindow, Ui_dialog, Ui_About

__all__ = ["start_gui"]


WINDOW_TITLE = "DiscoPygal Solver Viewer"
DEFAULT_ZOOM = 30


def get_available_solvers():
    """
    Return a list of all available solvers' names
    """
    solver_list = []
    for obj in globals():
        if isclass(globals()[obj]) and issubclass(globals()[obj], Solver) and globals()[obj] is not Solver:
            solver_list.append(obj)
    return solver_list


class MESSAGE_TYPE(Enum):
    INFO = 0,
    ERROR = 1


def pop_message_box(type, title, message):
    ICONS = {
        MESSAGE_TYPE.INFO: QtWidgets.QMessageBox.Icon.Information,
        MESSAGE_TYPE.ERROR: QtWidgets.QMessageBox.Icon.Critical
    }

    msgbox = QtWidgets.QMessageBox(ICONS[type], title, message)
    msgbox.exec()


def import_solver_file(path):
    try:
        sys.path.append(os.path.dirname(path))
        module_name = os.path.basename(path).rstrip(".py")
        if module_name in sys.modules:
            module = sys.modules[module_name]
            importlib.reload(module)
        else:
            module = importlib.import_module(module_name)
        cnt = 0
        for obj_name in dir(module):
            obj = getattr(module, obj_name)
            if isclass(obj) and issubclass(obj, Solver) and obj_name != "Solver":
                globals()[obj_name] = obj
                cnt += 1
        pop_message_box(MESSAGE_TYPE.INFO,
                        "Import solvers",
                        "Successfully import {} solvers from {}.".format(cnt, path))
    except Exception as e:
        message = f"Exception {e}\n{traceback.format_exc()}"
        print(message)
        pop_message_box(MESSAGE_TYPE.ERROR, "Could not import module", message)


class SolverDialog(Ui_dialog):
    DEFAULT_SOLVER_TEXT = "Select a solver..."

    def __init__(self, gui, dialog):
        super().__init__()
        self.gui = gui  # pointer to parent gui
        self.dialog = dialog
        self.setupUi(self.dialog)

        self.update_combo_box()
        self.selectButton.clicked.connect(self.choose_solver)
        self.browseButton.clicked.connect(self.solver_from_file)

    def solver_from_file(self):
        """
        Choose a solver from a file
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.dialog, 'Load File')
        if path == '':
            return
        import_solver_file(path)
        self.update_combo_box()

    def choose_solver(self):
        if self.DEFAULT_SOLVER_TEXT == self.solverComboBox.currentText():
            return
        self.gui.select_solver_class(self.solverComboBox.currentText())
        self.dialog.close()

    def update_combo_box(self):
        items = [self.DEFAULT_SOLVER_TEXT] + get_available_solvers()
        self.solverComboBox.clear()
        self.solverComboBox.addItems(items)


class SolverViewerGUI(Ui_MainWindow, GUI):
    def __init__(self):
        super().__init__()
        self.set_program_name(WINDOW_TITLE)

        # Fix initial zoom
        self.zoom = DEFAULT_ZOOM
        self.redraw()

        # Disable scene path edit
        self.scenePathEdit.setEnabled(False)

        # Setup the scene
        self.discopygal_scene = Scene()
        self.scene_drawer = SceneDrawer(self, self.discopygal_scene)
        self.scene_path = ""
        self.actionOpen_Scene.triggered.connect(self.choose_scene)
        self.actionOpenScene.triggered.connect(self.choose_scene)
        self.actionClear.triggered.connect(self.clear)

        # Setup solver
        self.solver = None
        self.solver_gui_elements = {}
        self.solver_graph = None
        self.solver_arrangement = None
        self.solver_graph_vertices = [] # gui
        self.solver_graph_edges = [] # gui
        self.actionOpenSolver.triggered.connect(self.open_solver_dialog)
        self.actionOpen_Solver.triggered.connect(self.open_solver_dialog)
        self.actionSolve.triggered.connect(self.solve)

        # Setup concurrency
        self.threadpool = QtCore.QThreadPool()
        self.worker = None
        self.logger = Logger(self.textEdit)
        self.writer = Writer(self.logger)
        self.start_time = None

        # Solution paths and misc data
        self.paths = None
        self.path_vertices = []  # gui
        self.path_edges = []  # gui
        self.set_animation_finished_action(self.anim_finished)
        self.bounding_box_edges = [] # gui
        # Setup actions
        self.actionShowPaths.triggered.connect(self.toggle_paths)
        self.actionPlay.triggered.connect(self.animate_paths)
        self.actionPause.triggered.connect(self.action_pause)
        self.actionStop.triggered.connect(self.action_stop)
        self.actionShowGraph.triggered.connect(self.action_display_graph)
        self.actionShowArrangement.triggered.connect(self.action_display_arrangement)
        self.actionAbout.triggered.connect(self.about_dialog)
        # self.actionQuit.triggered.connect(lambda: sys.exit(0)) # Raises an error of leaked instances
        self.actionVerify.triggered.connect(self.verify_paths)
        self.actionShowBoundingBox.triggered.connect(self.toggle_bounding_box)

    def setupUi(self):
        super().setupUi(self.mainWindow)

    def verify_paths(self):
        """
        Verify paths action
        """
        res, reason = verify_paths(self.discopygal_scene, self.paths)
        if res:
            pop_message_box(MESSAGE_TYPE.INFO, "Verify paths", "Successfully verified paths.")
        else:
            pop_message_box(MESSAGE_TYPE.ERROR, "Verify paths", "Paths are invalid: " + reason)

    def about_dialog(self):
        """
        Open the about dialog
        """
        dialog = QtWidgets.QDialog()
        dialog.ui = Ui_About()
        dialog.ui.setupUi(dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.setWindowTitle('About')
        dialog.exec_()

    def action_display_arrangement(self):
        """
        Display arrangement, if applicable
        """
        if self.solver_arrangement is None:
            return
        display_arrangement(self.solver_arrangement)
    def action_display_graph(self):
        """
        Display graph, if applicable
        """
        if len(self.solver_graph_vertices) > 0:
            self.clear_graph()
        else:
            self.show_graph()
    def show_graph(self):
        """
        Display the solver graph
        """
        if self.solver_graph is None:
            return

        nodes = list(self.solver_graph.nodes(data=True))
        nodes = [node[0] for node in nodes]

        for node in nodes:
            if type(node) is Point_2:
                x1, y1 = node.x().to_double(), node.y().to_double()

                self.solver_graph_vertices.append(
                    self.add_disc(
                        0.05, x1, y1, QtCore.Qt.red, QtCore.Qt.red
                    )
                )
            else:
                for i in range(node.dimension() // 2):
                    x1, y1 = node[2*i], node[2*i+1]
                    self.solver_graph_vertices.append(
                        self.add_disc(
                            0.05, x1, y1, QtCore.Qt.red, QtCore.Qt.red
                        )
                    )

    def clear_graph(self):
        """
        If a graph is drawn, clear it
        """
        for vertex in self.solver_graph_vertices:
            self.scene.removeItem(vertex.disc)
        for edge in self.solver_graph_edges:
            self.scene.removeItem(edge.line)
        self.solver_graph_vertices.clear()
        self.solver_graph_edges.clear()

    def action_pause(self):
        """
        Pause animation button
        """
        if self.is_queue_playing():
            self.pause_queue()

    def action_stop(self):
        """
        Stop the animation
        """
        if self.is_queue_playing() or self.is_queue_paused():
            self.stop_queue()
        # Move robots back to start
        self.scene_drawer.clear_scene()
        self.scene_drawer.draw_scene()

    def toggle_paths(self):
        """
        Toggle paths button
        """
        if len(self.path_vertices) > 0:
            self.clear_paths()
        else:
            self.draw_paths()

    def anim_finished(self):
        """
        This is called when the animation is finished
        """
        pass

    def animate_paths(self):
        """
        Animate the paths (if exists)
        """
        if self.paths is None or len(self.paths.paths) == 0 or self.is_queue_playing():
            return

        # If we are just paused, resume
        if self.is_queue_paused():
            self.play_queue()
            return

        # Otherwise generate the paths for animation
        self.scene_drawer.clear_scene()
        self.scene_drawer.draw_scene()

        path_len = len(list(self.paths.paths.values())[0].points)  # num of edges in paths
        animations = []
        for i in range(path_len-1):
            # All robots move in parallel along their edge
            animation_edges = []
            for robot in self.paths.paths:
                robot_gui = self.scene_drawer.robot_lut[robot][0]
                source = self.paths.paths[robot].points[i]
                target = self.paths.paths[robot].points[i+1]
                if 'theta' in source.data:
                    itheta = source.data['theta']
                    theta = target.data['theta']
                    if type(itheta) is not float:
                        itheta = itheta.to_double()
                    if type(theta) is not float:
                        theta = theta.to_double()
                    clockwise = source.data['clockwise']
                ix = source.location.x().to_double()
                iy = source.location.y().to_double()
                x = target.location.x().to_double()
                y = target.location.y().to_double()
                if 'theta' not in source.data:
                    animation_edges.append(
                        self.linear_translation_animation(
                            robot_gui, ix, iy, x, y, 250
                        )
                    )
                else:
                    animation_edges.append(
                        self.segment_angle_animation(
                            robot_gui, ix, iy, itheta, x, y, theta, clockwise, 500
                        )
                    )
            animations.append(self.parallel_animation(*animation_edges))

        self.queue_animation(*animations)
        self.play_queue()

    def draw_paths(self):
        """
        Draw the paths (if exist)
        """
        if self.paths is None:
            return

        for robot in self.paths.paths:
            points = self.paths.paths[robot].points
            for i in range(len(points)):
                x1, y1 = points[i].location.x().to_double(
                ), points[i].location.y().to_double()
                self.path_vertices.append(self.add_disc(
                    0.05, x1, y1, QtCore.Qt.darkGreen, QtCore.Qt.darkGreen))

                if i < len(points)-1:
                    if points[i] == points[i+1]:
                        continue
                    x2, y2 = points[i+1].location.x().to_double(), points[i + 1].location.y().to_double()
                    self.path_edges.append(self.add_segment(
                        x1, y1, x2, y2, QtCore.Qt.darkGreen))

    def clear_paths(self):
        """
        Clear the paths if any were drawn
        """
        for vertex in self.path_vertices:
            self.scene.removeItem(vertex.disc)
        for edge in self.path_edges:
            self.scene.removeItem(edge.line)
        self.path_vertices.clear()
        self.path_edges.clear()

    def get_solver_args(self):
        """
        Extract a dict from the dynamically generated GUI arguments (to pass to the solver)
        """
        args = {}
        solver_args = self.solver.get_arguments()
        for arg in self.solver_gui_elements:
            if arg.endswith('_label'):
                continue
            _, _, ttype = solver_args[arg]
            args[arg] = ttype(self.solver_gui_elements[arg].text())
        return args

    @staticmethod
    def add_padding_to_paths(path_collection):
        """
        Add more points (end points) to the paths that are shorter than the maximum path so all paths
        will be at the same size
        """
        if path_collection is None or not path_collection.paths:
            return path_collection
        max_length = max((len(path.points) for path in path_collection.paths.values()))
        padded_path_collection = PathCollection()
        for robot, path in path_collection.paths.items():
            path.points.extend([PathPoint(robot.end) for _ in range(max_length - len(path.points))])
            padded_path_collection.add_robot_path(robot, path)

        return padded_path_collection

    def solve_thread(self):
        """
        The thread that is run by the "solve" function"
        """
        args = self.get_solver_args()
        self.solver = self.solver.__class__(**args) # Create a new solver object for solving
        self.solver.set_verbose(self.writer)
        try:
            self.solver.load_scene(self.discopygal_scene)
            self.paths = self.add_padding_to_paths(self.solver.solve())
        except Exception as e:
            print("Error in solving scene", file=self.writer)
            print(f"Exception: {e}", file=self.writer)
            traceback.print_exc()
            return
        self.solver_graph = self.solver.get_graph()
        self.solver_arrangement = self.solver.get_arrangement()

    def solve(self):
        """
        This method is called by the solve button.
        Run the MP solver in parallel to the app.
        """
        if self.solver is None:
            return
        self.disable_toolbar()
        self.worker = Worker(self.solve_thread)
        self.worker.signals.finished.connect(self.solver_done)
        self.threadpool.start(self.worker)
        self.start_time = time.time()

    def disable_toolbar(self):
        """
        Disable icons on the toolbar while running
        """
        self.toolBar.setEnabled(False)

    def solver_done(self):
        """
        Enable icons on the toolbar after done running
        """
        self.toolBar.setEnabled(True)
        end_time = time.time()
        print("Time took: {}[sec]".format(end_time - self.start_time), file=self.writer)

    def select_solver_class(self, solver_name):
        self.load_solver(globals()[solver_name].init_default_solver())

    def load_solver(self, solver):
        """
        Set the selected solver.
        Also generate dynamically the GUI elements corresponding to the solver's arguments.
        """

        self.solver = solver

        # Clear all elements
        for element in self.solver_gui_elements.values():
            element.setParent(None)
        self.solver_gui_elements.clear()

        # Generate the settings layout
        layout = QtWidgets.QVBoxLayout()
        args = self.solver.get_arguments()
        for arg, (description, default, _) in args.items():
            self.solver_gui_elements[arg + '_label'] = QtWidgets.QLabel(description)
            layout.addWidget(self.solver_gui_elements[arg + '_label'])
            self.solver_gui_elements[arg] = QtWidgets.QLineEdit(str(self.solver.__getattribute__(arg)))
            layout.addWidget(self.solver_gui_elements[arg])
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        # Attach layout to scroll widget
        self.scrollArea.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(widget)
        self.solverName.setText(type(solver).__name__)

    def update_scene_metadata(self):
        """
        Update the scene's metadata
        """
        if 'version' in self.discopygal_scene.metadata:
            self.versionEdit.setText(self.discopygal_scene.metadata['version'])
        else:
            self.versionEdit.setText('NO VERSION')

        if 'solvers' in self.discopygal_scene.metadata:
            self.solversEdit.setText(self.discopygal_scene.metadata['solvers'])
        else:
            self.solversEdit.setText('NO SOLVERS')

        if 'details' in self.discopygal_scene.metadata:
            self.sceneDetailsEdit.setPlainText(self.discopygal_scene.metadata['details'])
        else:
            self.sceneDetailsEdit.setPlainText('NO DETAILS')

    def open_solver_dialog(self):
        """
        Open the "load solver" dialog
        """
        dialog = QtWidgets.QDialog()
        dialog.ui = SolverDialog(self, dialog)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        dialog.setWindowTitle('Open Solver...')
        dialog.exec_()

    def choose_scene(self):
        """
        Load a scene.
        """
        name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.mainWindow, 'Load File')
        if name == '':
            return

        self.load_scene(name)

    def load_scene(self, scene):
        self.clear()
        if not isinstance(scene, Scene):
            scene_file = scene
            try:
                with open(scene_file, 'r') as fp:
                    d = json.load(fp)
                    scene = Scene.from_dict(d)
                self.scene_path = scene_file
                self.scenePathEdit.setText(self.scene_path)
            except FileNotFoundError:
                pop_message_box(MESSAGE_TYPE.ERROR,
                                "Scene file error",
                                f"File {scene_file} not found!")
                return
            except (KeyError, json.decoder.JSONDecodeError) as e:
                pop_message_box(MESSAGE_TYPE.ERROR,
                                "Scene file load failure",
                                f"Failed to load scene {scene_file}\n"
                                f"Error: {e}")
                return

        self.discopygal_scene = scene
        self.scene_drawer.clear_scene()
        self.scene_drawer.scene = self.discopygal_scene
        self.scene_drawer.draw_scene()
        self.update_scene_metadata()

        self.clear_paths()
        if self.paths is not None:
            self.paths.paths.clear()
        self.paths = None

    def clear(self):
        """
        Clear scene, paths, graphs, etc.
        """
        self.scene_drawer.clear_scene()

        self.discopygal_scene = Scene()
        self.scene_drawer.scene = self.discopygal_scene
        self.scene_drawer.draw_scene()
        self.scene_path = ""
        self.clear_graph()
        self.clear_paths()
        self.clear_bounding_box()
        if self.paths is not None:
            self.paths.paths.clear()
        self.paths = None
        self.scenePathEdit.setText(self.scene_path)

    def toggle_bounding_box(self):
        if self.bounding_box_edges:
            self.clear_bounding_box()
        else:
            self.draw_bounding_box()


    def draw_bounding_box(self):
        if self.solver is None:
            return

        bounding_box_graph = self.solver.get_bounding_box_graph()

        if bounding_box_graph is None:
            return

        for edge in bounding_box_graph.edges:
            p, q = edge
            x1, y1 = p.x().to_double(), p.y().to_double()
            x2, y2 = q.x().to_double(), q.y().to_double()
            self.bounding_box_edges.append(
                self.add_segment(x1, y1, x2, y2, QtCore.Qt.gray, opacity=0.7))

    def clear_bounding_box(self):
        for edge in self.bounding_box_edges:
            self.scene.removeItem(edge.line)

        self.bounding_box_edges.clear()

    def exit(self):
        self.clear()
        sys.exit(0)


def start_gui(scene=None, solver=None, solver_file=None, graph=None):
    """
    Start solver_viewer tool

    See also :ref:`Solver Viewer - From script <solver_viewer_script>`

    :param scene: scene to upload
    :type scene: :class:`~discopygal.solvers.Scene`

    :param solver: solver to upload.
         May be a solver object (object of a class that inherits from :class:`~discopygal.solvers.Solver`),
         class that inherits from :class:`~discopygal.solvers.Solver` or a name of a solver's class.

    :type solver: :class:`~discopygal.solvers.Solver` or :class:`class` or :class:`str`
    """
    app = QtWidgets.QApplication(sys.argv)
    gui = SolverViewerGUI()
    gui.mainWindow.show()
    gui.solver_graph = graph
    gui.action_display_graph()

    if scene is not None:
        gui.load_scene(scene)
    if solver_file is not None:
        import_solver_file(solver_file)
    if solver is not None:
        if isinstance(solver, Solver):
            gui.load_solver(solver)
        elif isclass(solver):
            gui.load_solver(solver.init_default_solver())
        else:
            try:
                gui.select_solver_class(solver)
            except KeyError:
                pop_message_box(MESSAGE_TYPE.ERROR,
                                "Invalid solver",
                                f"Invalid solver name.\nValid solvers are: {get_available_solvers()}")

    app.exec_()

