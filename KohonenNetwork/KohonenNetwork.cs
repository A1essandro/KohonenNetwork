using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using KohonenNetwork.Learning;
using NeuralNetworkConstructor.Constructor;
using NeuralNetworkConstructor.Constructor.Generators;
using NeuralNetworkConstructor.Structure.ActivationFunctions;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;
using NeuralNetworkConstructor.Structure.Synapses;

namespace KohonenNetwork
{
    public class KohonenNetwork : TwoLayersNetwork
    {

        public KohonenNetwork(IReadOnlyLayer<IMasterNode> inputLayer, IReadOnlyLayer<INotInputNode> outputLayer)
            : base(inputLayer, outputLayer)
        {
        }

        /// <summary>
        /// Get prepeared result (values {0, 1})
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public override async Task<IEnumerable<double>> Output() => _prepareResult(await RawOutput().ConfigureAwait(false));

        /// <summary>
        /// Get unprepared result
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Task<IEnumerable<double>> RawOutput() => base.Output();

        /// <summary>
        /// Get index of neuron with maximum result output
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public async Task<int> GetOutputIndex() => _getWinnerIndex(await Output().ConfigureAwait(false));

        #region Private methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double[] _prepareResult(IEnumerable<double> raw)
        {
            var winnerIndex = _getWinnerIndex(raw);
            var result = new double[_outputLayer.Nodes.Count()];
            result[winnerIndex] = 1;

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int _getWinnerIndex(IEnumerable<double> raw) => Array.IndexOf(raw.ToArray(), raw.Max());

        #endregion

    }
}
