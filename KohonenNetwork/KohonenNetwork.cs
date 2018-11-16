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
    public class KohonenNetwork : TwoLayersNetwork
    {

        public KohonenNetwork(int inputNodes, int outputNodes, bool withBias = false)
            : this(new NetworkConfiguration(inputNodes, outputNodes, withBias))
        {
        }

        public KohonenNetwork(NetworkConfiguration config)
            : base(LayerGenerator.GenerateInputLayer(config.InputLayerNodes, config.CreateBiasNode), LayerGenerator.GenerateOutputLayer(config.OutputLayerNodes))
        {
            Synapse.Generator.EachToEach(InputLayer, OutputLayer, config.SynapseWeightGenerator);
        }

        public override async Task<IEnumerable<double>> Output() => _prepareResult(await RawOutput().ConfigureAwait(false));

        public Task<IEnumerable<double>> RawOutput() => base.Output();

        public async Task<int> GetOutputIndex() => _getWinnerIndex(await Output().ConfigureAwait(false));

        #region Private methods

        private double[] _prepareResult(IEnumerable<double> raw)
        {
            var winnerIndex = _getWinnerIndex(raw);
            var result = new double[_outputLayer.Nodes.Count];
            result[winnerIndex] = 1;

            return result;
        }

        private int _getWinnerIndex(IEnumerable<double> raw) => Array.IndexOf(raw.ToArray(), raw.Max());

        #endregion

    }
}
