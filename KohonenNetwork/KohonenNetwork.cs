using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KohonenNetwork.Learning;
using NeuralNetworkConstructor.Constructor;
using NeuralNetworkConstructor.Structure.ActivationFunctions;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;
using NeuralNetworkConstructor.Structure.Synapses;

namespace KohonenNetwork
{
    public class KohonenNetwork<TFunc> : TwoLayersNetwork
        where TFunc : IActivationFunction, new()
    {

        public KohonenNetwork(int inputNodes, int outputNodes, bool withBias = false)
            : this(new NetworkConfiguration(inputNodes, outputNodes, withBias))
        {
        }

        public KohonenNetwork(NetworkConfiguration config)
            : base(CreateAndGetInputLayer(config.InputLayerNodes, config.CreateBiasNode), CreateAndGetOutputLayer(config.OutputLayerNodes))
        {
            Synapse.Generator.EachToEach(InputLayer, OutputLayer, config.SynapseWeightGenerator);
        }

        public override async Task<IEnumerable<double>> Output() => _prepareResult(await RawOutput().ConfigureAwait(false));

        public Task<IEnumerable<double>> RawOutput() => base.Output();

        public async Task<int> GetOutputIndex()
        {
            var output = await Output().ConfigureAwait(false);

            return Array.IndexOf(output.ToArray(), output.Max());
        }

        #region Private methods

        private double[] _prepareResult(IEnumerable<double> raw)
        {
            var winnerIndex = Array.IndexOf(raw.ToArray(), raw.Max());
            var result = new double[_outputLayer.Nodes.Count];
            result[winnerIndex] = 1;

            return result;
        }

        private static InputLayer CreateAndGetInputLayer(int qty, bool withBias)
        {
            var result = new InputLayer();
            for (var i = 0; i < qty; i++)
            {
                result.Nodes.Add(new InputNode());
            }

            if (withBias)
            {
                result.Nodes.Add(new Bias());
            }

            return result;
        }

        private static Layer CreateAndGetOutputLayer(int qty)
        {
            var result = new Layer();
            for (var i = 0; i < qty; i++)
            {
                result.Nodes.Add(new Neuron());
            }

            return result;
        }

        #endregion

    }
}
