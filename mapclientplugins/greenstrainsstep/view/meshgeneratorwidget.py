"""
Created on Aug 29, 2017

@author: Richard Christie
"""

import types


import json
import webbrowser
import numpy as np

from PySide import QtGui, QtCore

from mapclient.view.utils import set_wait_cursor
from mapclientplugins.greenstrainsstep.view.strainwidget import Ui_MeshGeneratorWidget
from mapclientplugins.greenstrainsstep.model.strainmesh import StrainMesh
from mapclientplugins.greenstrainsstep.model.video import Video

class MeshGeneratorWidget(QtGui.QWidget):

    def __init__(self, model, node_coordinates_data, parent=None):
        super(MeshGeneratorWidget, self).__init__(parent)
        self._ui = Ui_MeshGeneratorWidget()
        self._model = model
        self._model.registerTimeValueUpdateCallback(self._updateTimeValue)
        self._model.registerFrameIndexUpdateCallback(self._updateFrameIndex)

        self._ui.setupUi(self)
        self._doneCallback = None
        self._marker_mode_active = False

        self.time = 0
        self.pw = None
        self.data = None
        self._node_coordinates_data = node_coordinates_data
        self._time_sequence = node_coordinates_data['time_array']

        self.video = Video(self._model.getVideoPath(), 30)
        self._ui.sceneviewer_widget.setContext(model.getContext())
        self._ui.sceneviewer_widget.setModel(self._model)
        self._ui.sceneviewer_widget.initializeGL()
        self._makeConnections()

        self._ui.sceneviewer_widget.grid = []

    def _graphicsInitialized(self):
        """
        Callback for when SceneviewerWidget is initialised
        Set custom scene from model
        """
        sceneviewer = self._ui.sceneviewer_widget.getSceneviewer()
        if sceneviewer is not None:
            self._refreshOptions()
            scene = self._model.getScene()
            self._ui.sceneviewer_widget.setScene(scene)
            # self._ui.sceneviewer_widget.setSelectModeAll()
            sceneviewer.setLookatParametersNonSkew([2.0, -2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
            sceneviewer.setTransparencyMode(sceneviewer.TRANSPARENCY_MODE_SLOW)
            self._autoPerturbLines()
            self._viewAll()

    def _sceneChanged(self):
        sceneviewer = self._ui.sceneviewer_widget.getSceneviewer()
        if sceneviewer is not None:
            if self._have_images:
                self._plane_model.setSceneviewer(sceneviewer)
            scene = self._model.getScene()
            self._ui.sceneviewer_widget.setScene(scene)
            self._autoPerturbLines()

    def _sceneAnimate(self):
        sceneviewer = self._ui.sceneviewer_widget.getSceneviewer()
        if sceneviewer is not None:
            self._model.loadSettings()
            scene = self._model.getScene()
            self._ui.sceneviewer_widget.setScene(scene)
            self._autoPerturbLines()
            self._viewAll()

    def _autoPerturbLines(self):
        """
        Enable scene viewer perturb lines iff solid surfaces are drawn with lines.
        Call whenever lines, surfaces or translucency changes
        """
        sceneviewer = self._ui.sceneviewer_widget.getSceneviewer()
        if sceneviewer is not None:
            #sceneviewer.setPerturbLinesFlag(self._generator_model.needPerturbLines())
            pass

    def _makeConnections(self):
        self._ui.sceneviewer_widget.graphicsInitialized.connect(self._graphicsInitialized)
        self._ui.done_button.clicked.connect(self._doneButtonClicked)
        self._ui.viewAll_button.clicked.connect(self._viewAll)
        self._ui.timeValue_doubleSpinBox.valueChanged.connect(self._timeValueChanged)
        self._ui.timePlayStop_pushButton.clicked.connect(self._timePlayStopClicked)
        self._ui.frameIndex_spinBox.valueChanged.connect(self._frameIndexValueChanged)
        self._ui.framesPerSecond_spinBox.valueChanged.connect(self._framesPerSecondValueChanged)
        self._ui.timeLoop_checkBox.clicked.connect(self._timeLoopClicked)
        self._ui.strain_reference_button.clicked.connect(self._set_strain_reference)
        self._ui.viewVideo_button.clicked.connect(self._playVideo)
        self._ui.view_mesh_button.clicked.connect(self._renderECGMesh)

    def _createFMAItem(self, parent, text, fma_id):
        item = QtGui.QTreeWidgetItem(parent)
        item.setText(0, text)
        item.setData(0, QtCore.Qt.UserRole + 1, fma_id)
        item.setCheckState(0, QtCore.Qt.Unchecked)
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsTristate)

        return item

    def getModel(self):
        return self._model

    def registerDoneExecution(self, doneCallback):
        self._doneCallback = doneCallback

    def _updateUi(self):
        pass

    def _doneButtonClicked(self):
        self._ui.dockWidget.setFloating(False)
        self._model.done()
        self._model = None
        self._doneCallback()


    def _updateTimeValue(self, value):
        self._ui.timeValue_doubleSpinBox.blockSignals(True)
        max_time_value = self.video.videoLength
        self.time = self._model._current_time


        if value > max_time_value:
            self._ui.timeValue_doubleSpinBox.setValue(max_time_value)
            self._timePlayStopClicked()
        else:
            self._ui.timeValue_doubleSpinBox.setValue(value)

        self._ui.timeValue_doubleSpinBox.blockSignals(False)

    def _renderECGMesh(self):

        self._pm = StrainMesh(self._model.get_region(), self._node_coordinates_data)
        self._pm.generate_mesh()
        self._pm.drawMesh()
        self._ui.sceneviewer_widget.setModel(self._pm)

    def _updateFrameIndex(self, value):
        self._ui.frameIndex_spinBox.blockSignals(True)
        self._ui.frameIndex_spinBox.setValue(value)
        self._ui.frameIndex_spinBox.blockSignals(False)

    def _timeValueChanged(self, value):
        self._model.setTimeValue(value)
        # self._pm.display_strains_at_given_time(int(value))
        # self._pm.display_strains_at_given_time_to_reference(int(value))

    def _timeDurationChanged(self, value):
        self._model.setTimeDuration(value)

    def _timePlayStopClicked(self):
        play_text = 'Play'
        stop_text = 'Stop'
        current_text = self._ui.timePlayStop_pushButton.text()
        if current_text == play_text:
            self._ui.timePlayStop_pushButton.setText(stop_text)
            self._model.play()
        else:
            self._ui.timePlayStop_pushButton.setText(play_text)
            self._model.stop()

    def _timeLoopClicked(self):
        self._model.setTimeLoop(self._ui.timeLoop_checkBox.isChecked())

    def _frameIndexValueChanged(self, value):
        self._model.setFrameIndex(value)

    def _framesPerSecondValueChanged(self, value):
        self._model.setFramesPerSecond(value)
        self._ui.timeValue_doubleSpinBox.setMaximum(self.video.numFrames/value)

    def _set_strain_reference(self):
         if self._pm is not None and self.vid is not None:
            self._pm.set_strain_reference_frame(self.vid.frameCount)

    def _playVideo(self):
        self.vid = Video(self._model.getVideoPath(), 30)
        if self.data is not None:
            self._adjustData()
            self.vid.line = self.plot.line
            self.vid.datalen = self.plot.datalen
        self.vid.playVideo()

    def _refreshOptions(self):
        self._ui.framesPerSecond_spinBox.setValue(self._model.getFramesPerSecond())
        self._ui.timeLoop_checkBox.setChecked(self._model.isTimeLoop())

    def _annotationItemChanged(self, item):
        print(item.text(0))
        print(item.data(0, QtCore.Qt.UserRole + 1))

    def _viewAll(self):
        """
        Ask sceneviewer to show all of scene.
        """
        if self._ui.sceneviewer_widget.getSceneviewer() is not None:
            self._ui.sceneviewer_widget.viewAll()

    def keyPressEvent(self, event):
        if event.modifiers() & QtCore.Qt.CTRL and QtGui.QApplication.mouseButtons() == QtCore.Qt.NoButton:
            self._marker_mode_active = True

            self._ui.sceneviewer_widget._calculatePointOnPlane = types.MethodType(_calculatePointOnPlane, self._ui.sceneviewer_widget)
            self._ui.sceneviewer_widget.mousePressEvent = types.MethodType(mousePressEvent, self._ui.sceneviewer_widget)
            self._model.printLog()

    def keyReleaseEvent(self, event):
        if self._marker_mode_active:
            self._marker_mode_active = False
            self._ui.sceneviewer_widget._calculatePointOnPlane = None
            self._ui.sceneviewer_widget.mousePressEvent = self._original_mousePressEvent


def mousePressEvent(self, event):
    if self._active_button != QtCore.Qt.NoButton:
        return

    if (event.modifiers() & QtCore.Qt.CTRL) and event.button() == QtCore.Qt.LeftButton:
        point_on_plane = self._calculatePointOnPlane(event.x(), event.y())
        print('Location of click (x,y): (' + str(event.x()) + ', ' + str(event.y()) +')')
        node = self.getNearestNode(event.x(), event.y())
        if node.isValid():
            print('node {0} was clicked'.format(node.getIdentifier()))
            self.foundNode = True
            self.nodeKey = node.getIdentifier()
            self.node = node
            self.grid = []

        # return sceneviewers 'mouspressevent' function to its version for navigation
        self._calculatePointOnPlane = None
        self.mousePressEvent = self.original_mousePressEvent

    return [event.x(), event.y()]


def _calculatePointOnPlane(self, x, y):
    from opencmiss.utils.maths.algorithms import calculateLinePlaneIntersection

    far_plane_point = self.unproject(x, -y, -1.0)
    near_plane_point = self.unproject(x, -y, 1.0)
    plane_point, plane_offset, plane_normal = self._model.getPlaneDescription()
    point_on_plane = calculateLinePlaneIntersection(near_plane_point, far_plane_point, plane_point, plane_normal)
    # if len(self.grid) < 4:
    #     self.grid.append(point_on_plane)
    # else:
    #     self.grid = []
    #     self.grid.append(point_on_plane)
    return point_on_plane
