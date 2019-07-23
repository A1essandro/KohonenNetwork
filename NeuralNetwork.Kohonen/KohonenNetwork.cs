using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen
{
    public class KohonenNetwork<TLayer> : TwoLayersNetwork<TLayer>, IKohonenNetwork
        where TLayer : IReadOnlyLayer<INotInputNode>
    {

        public KohonenNetwork(IReadOnlyLayer<IMasterNode> inputLayer, TLayer outputLayer)
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
        public async Task<int?> GetOutputIndex() => _getWinnerIndex(await Output().ConfigureAwait(false));

        #region Private methods

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private double[] _prepareResult(IEnumerable<double> raw)
        {
            var winnerIndex = _getWinnerIndex(raw);
            var result = new double[OutputLayer.Nodes.Count()];
            if (winnerIndex.HasValue)
            {
                result[winnerIndex.Value] = 1;
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int? _getWinnerIndex(IEnumerable<double> raw)
        {
            return raw.Count() > 0
                ? Array.IndexOf(raw.ToArray(), raw.Max())
                : new int?();
        }

        #endregion

    }

    public class KohonenNetwork : KohonenNetwork<IReadOnlyLayer<INotInputNode>>
    {

        public KohonenNetwork(IReadOnlyLayer<IMasterNode> inputLayer, IReadOnlyLayer<INotInputNode> outputLayer)
            : base(inputLayer, outputLayer)
        {
        }

    }

}
