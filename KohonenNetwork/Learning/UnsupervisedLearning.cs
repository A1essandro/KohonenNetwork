using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Common;
using NeuralNetworkConstructor.Networks;
using NeuralNetworkConstructor.Structure.Nodes;

namespace KohonenNetwork.Learning
{

    /// <summary>
    /// Class for learning network
    /// </summary>
    public class UnsupervisedLearning : IUnsupervisedLearning
    {

        private Network _network;
        private readonly LearningConfiguration _config;

        public UnsupervisedLearning(Network network, LearningConfiguration config = null)
        {
            _network = network;
            _config = config;
        }

        public UnsupervisedLearning(Network network, double force, IOrganizing organizingAlgorithm = null)
            : this(network, new LearningConfiguration(force, organizingAlgorithm))
        {
        }

        public async Task Learn(IEnumerable<double> input)
        {
            if (_config.OrganizingAlgorithm != null && await _config.OrganizingAlgorithm.Organize(input))
            {
                return;
            }

            _network.Input(input);
            _recalcWeights(await _network.Output());
        }

        public async Task Learn(IEnumerable<IEnumerable<double>> epoch, int? repeats = null)
        {
            if (!repeats.HasValue)
            {
                repeats = _config.DefaultRepeatsNumber;
            }

            var random = new Random();
            var initialTheta = _config.Theta;

            for (var i = 0; i < repeats.Value; i++)
            {
                if (_config.ShuffleEveryEpoch)
                {
                    epoch = epoch.OrderBy(a => random.NextDouble());
                }

                foreach (var input in epoch)
                {
                    await Learn(input);
                }

                _config.Theta *= _config.ThetaFactorPerEpoch;
            }
            _config.Theta = initialTheta;
        }

        #region private methods

        private void _recalcWeights(IEnumerable<double> output)
        {
            _getWinner(output).Synapses.AsParallel().ForAll(async synapse =>
            {
                var nodeOutput = await synapse.MasterNode.Output().ConfigureAwait(false);
                synapse.ChangeWeight(_config.Theta * (nodeOutput - synapse.Weight));
            });
        }

        private ISlaveNode _getWinner(IEnumerable<double> output)
        {
            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            return _network.OutputLayer.Nodes.ToArray()[winnerIndex] as ISlaveNode;
        }

        #endregion

    }
}