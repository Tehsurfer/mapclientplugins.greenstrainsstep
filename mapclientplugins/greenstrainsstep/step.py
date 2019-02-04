
"""
MAP Client Plugin Step
"""
from PySide import QtGui

from mapclient.mountpoints.workflowstep import WorkflowStepMountPoint
from mapclientplugins.greenstrainsstep.configuredialog import ConfigureDialog
from mapclientplugins.greenstrainsstep.view.meshgeneratorwidget import MeshGeneratorWidget
from mapclientplugins.greenstrainsstep.model.mastermodel import MasterModel

class GreenStrainsStep(WorkflowStepMountPoint):
    """
    Skeleton step which is intended to be a helpful starting point
    for new steps.
    """

    def __init__(self, location):
        super(GreenStrainsStep, self).__init__('GreenStrains', location)
        self._configured = True # A step cannot be executed until it has been configured.
        self._category = 'Fitting'
        # Add any other initialisation code here:
        self._icon =  QtGui.QImage(':/greenstrainsstep/images/stretch-2-65711.png')
        # Ports:
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#time_based_electrode_scaffold_positions'))
        self.addPort(('http://physiomeproject.org/workflow/1.0/rdf-schema#port',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#uses',
                      'http://physiomeproject.org/workflow/1.0/rdf-schema#file_location'))
        # Port data:
        self._portData0 = None # time_based_electrode_scaffold_positions
        self._portData1 = None # Video location

        # Config:
        self._config = {'identifier': ''}

    def execute(self):
        """
        Add your code here that will kick off the execution of the step.
        Make sure you call the _doneExecution() method when finished.  This method
        may be connected up to a button in a widget for example.
        """
        # Put your execute step code here before calling the '_doneExecution' method.
        self._model = MasterModel(self._location, self._config['identifier'], self._portData1)
        self._view = MeshGeneratorWidget(self._model, self._portData0)
        self._setCurrentWidget(self._view)
        self._view.registerDoneExecution(self._my_done_execution)

    def _my_done_execution(self):
        self._portData1 = self._model.getOutputModelFilename()
        self._view = None
        self._model = None
        self._doneExecution()

    def setPortData(self, index, dataIn):
        """
        Add your code here that will set the appropriate objects for this step.
        The index is the index of the port in the port list.  If there is only one
        uses port for this step then the index can be ignored.

        :param index: Index of the port to return.
        :param dataIn: The data to set for the port at the given index.
        """
        if index == 0:
            self._portData0 = dataIn # time_based_electrode_scaffold_positions
        if index == 1:
            self._portData1 = dataIn

    def configure(self):
        """
        This function will be called when the configure icon on the step is
        clicked.  It is appropriate to display a configuration dialog at this
        time.  If the conditions for the configuration of this step are complete
        then set:
            self._configured = True
        """
        dlg = ConfigureDialog(QtGui.QApplication.activeWindow().currentWidget())
        dlg.identifierOccursCount = self._identifierOccursCount
        dlg.setConfig(self._config)
        dlg.validate()
        dlg.setModal(True)

        if dlg.exec_():
            self._config = dlg.getConfig()

        self._configured = dlg.validate()
        self._configuredObserver()

    def getIdentifier(self):
        """
        The identifier is a string that must be unique within a workflow.
        """
        return self._config['identifier']

    def setIdentifier(self, identifier):
        """
        The framework will set the identifier for this step when it is loaded.
        """
        self._config['identifier'] = identifier

    def serialize(self):
        """
        Add code to serialize this step to string.  This method should
        implement the opposite of 'deserialize'.
        """
        pass

    def deserialize(self, string):
        """
        Add code to deserialize this step from string.  This method should
        implement the opposite of 'serialize'.

        :param string: JSON representation of the configuration in a string.
        """
        pass

