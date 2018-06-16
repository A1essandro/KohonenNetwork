using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KohonenNetwork.Learning;
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
        }

        public KohonenNetwork<TFunc> SetLearning(ISelfLearning learningAlgorithm)
        {
            _learning = learningAlgorithm;

            return this;
        }

        public void Learn(IEnumerable<double> input)
        {
            _learning.Learn(input);
        }

        public void Learn(ICollection<IEnumerable<double>> epoch, bool shuffle = true)
        {
            if (shuffle)
            {
                var random = new Random();
                epoch = epoch.OrderBy(a => random.NextDouble()).ToArray();
            }

            foreach (var input in epoch)
            {
                _learning.Learn(input);
            }
        }

        public override IEnumerable<double> Output()
        {
            var rawResult = base.Output();

            return _prepareResult(rawResult);
        }

        public override async Task<IEnumerable<double>> OutputAsync()
        {
            var rawResult = await base.OutputAsync();

            return _prepareResult(rawResult);
        }

        public int GetOutputIndex()
        {
            var output = base.Output();

            return Array.IndexOf(output.ToArray(), output.Max());
        }

        private double[] _prepareResult(IEnumerable<double> raw)
        {
            var winnerIndex = Array.IndexOf(raw.ToArray(), raw.Max());
            var result = new double[_outputLayer.Nodes.Count];
            result[winnerIndex] = 1;

            return result;
        }

    }
}
