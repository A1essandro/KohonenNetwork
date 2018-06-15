using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Constructor;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Layer;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;

namespace KohonenNetwork
{
    public class KohonenNetwork : INetwork
    {

        private const string OUTPUT_LAYER_ID = "output_layer";

        private readonly INetwork _network;

        public KohonenNetwork(int inputNodes, int outputNodes, bool withBias = true)
        {
            var constructor = new NetworkConstructor<Network>();

            for (var i = 0; i < inputNodes; i++)
            {
                constructor.AddInputNode<InputNode>(Guid.NewGuid().ToString());
            }
            if (withBias)
            {
                constructor.AddInputNode<InputBias>(Guid.NewGuid().ToString());
            }
            constructor.AddLayer<Layer>(OUTPUT_LAYER_ID);
            for (var i = 0; i < outputNodes; i++)
            {
                constructor.AddNeuron<Neuron<Gaussian>>(Guid.NewGuid().ToString());
            }

            _network = constructor.Complete();
        }

        public IInputLayer InputLayer => _network.InputLayer;

        public ICollection<ILayer<INode>> Layers => _network.Layers.ToArray();

        public ILayer<INode> OutputLayer => _network.OutputLayer;

        public event Action<IEnumerable<double>> OnOutput;
        public event Action<IEnumerable<double>> OnInput;

        public void Input(IEnumerable<double> input)
        {
            OnInput?.Invoke(input);
            _network.Input(input);
        }

        public IEnumerable<double> Output()
        {
            var result = _network.Output();
            OnOutput?.Invoke(result);

            return result;
        }

        public async Task<IEnumerable<double>> OutputAsync()
        {
            var result = await OutputAsync();
            OnOutput?.Invoke(result);

            return result;
        }

        public void Refresh()
        {
            _network.Refresh();
        }
    }
}
