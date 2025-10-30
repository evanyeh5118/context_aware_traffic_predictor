from ...models.ContextAssisted.TrafficPredictorEnhanced import TrafficPredictorContextAssisted, CustomLossFunction
from ...utils.device_utils import get_device


def createModel(parameters):
    """Create model with proper device handling."""
    len_source = parameters['len_source']
    len_target = parameters['len_target']
    num_classes = parameters['num_classes']
    input_size = parameters['input_size']
    output_size = parameters['output_size']
    hidden_size = parameters['hidden_size']
    num_layers = parameters['num_layers']
    dropout_rate = parameters['dropout_rate']
    dt = parameters['dt']
    degree = parameters['degree']

    device = get_device()

    model = TrafficPredictorContextAssisted(
        input_size, hidden_size, output_size, num_classes, len_source, len_target,
        dt, degree, device, num_layers=num_layers, dropout_rate=dropout_rate
    )

    return model, device


