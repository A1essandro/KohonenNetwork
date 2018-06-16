using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Constructor;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Layer;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using NeuralNetworkConstructor.Network.Node.Synapse;

namespace KohonenNetwork
{
    public class KohonenNetwork<TFunc> : TwoLayersNetwork
        where TFunc : IActivationFunction, new()
    {

        private ISelfLearning _learning;
        private ISelfOrganizing _organizing;

        public KohonenNetwork(int inputNodes, int outputNodes, bool withBias = true)
        {
            for (var i = 0; i < inputNodes; i++)
            {
                _inputLayer.Nodes.Add(new InputNode());
            }

            if (withBias)
            {
                _inputLayer.Nodes.Add(new InputBias());
            }

            for (var i = 0; i < inputNodes; i++)
            {
                _outputLayer.Nodes.Add(new Neuron<TFunc>());
            }

            Synapse.Generator.EachToEach(_inputLayer, _outputLayer);
            _learning = new SelfLearning<TFunc>(this);
            _organizing = new SelfOrganizing<TFunc>(this, 1);
        }

        public KohonenNetwork<TFunc> SetLearning(ISelfLearning learningAlgorithm)
        {
            _learning = learningAlgorithm;

            return this;
        }

        public KohonenNetwork<TFunc> SetOrganizing(ISelfOrganizing organizingAlgoritgm)
        {
            _organizing = organizingAlgoritgm;

            return this;
        }

        public int GetOutputIndex()
        {
            var output = Output();

            return output.Select((o, idx) => new { o, idx }).First(x => x.o == 1).idx;
        }

    }
}
