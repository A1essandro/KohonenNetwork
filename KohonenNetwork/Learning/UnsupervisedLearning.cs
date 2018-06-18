using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Network;
using NeuralNetworkConstructor.Network.Node;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;

namespace KohonenNetwork.Learning
{

    /// <summary>
    /// Class for learning network
    /// </summary>
    public class UnsupervisedLearning : IUnsupervisedLearning
    {

        private INetwork _network;
        private readonly LearningConfiguration _config;

        public UnsupervisedLearning(INetwork network, LearningConfiguration config = null)
        {
            _network = network;
            _config = config;
        }

        public UnsupervisedLearning(INetwork network, double force, IOrganizing organizingAlgorithm = null)
            : this(network, new LearningConfiguration(force, organizingAlgorithm))
        {
        }

        public void Learn(IEnumerable<double> input)
        {
            if (_config.OrganizingAlgorithm != null && _config.OrganizingAlgorithm.Organize(input))
            {
                return;
            }

            _network.Input(input);
            _recalcWeights(_network.Output());
        }

        public void Learn(IEnumerable<IEnumerable<double>> epoch, int? repeats = null)
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
                    Learn(input);
                }

                _config.Theta *= _config.ThetaFactorPerEpoch;
            }
            _config.Theta = initialTheta;
        }

        public async Task LearnAsync(IEnumerable<double> input)
        {
            if (_config.OrganizingAlgorithm != null
                && await _config.OrganizingAlgorithm.OrganizeAsync(input).ConfigureAwait(false))
            {
                return;
            }

            _network.Input(input);
            _recalcWeights(await _network.OutputAsync().ConfigureAwait(false)); //OutputAsync() has problem with performance
        }

        public async Task LearnAsync(IEnumerable<IEnumerable<double>> epoch, int? repeats = null)
        {
            //TODO: method _network.OutputAsync() has problem with performance, so just temporarily wrap to the task sync method
            await Task.Run(() => Learn(epoch, repeats));
        }

        #region private methods

        private void _recalcWeights(IEnumerable<double> output)
        {
            _getWinner(output).Synapses.AsParallel().ForAll(async synapse =>
            {
                var nodeOutput = await synapse.MasterNode.OutputAsync().ConfigureAwait(false);
                synapse.ChangeWeight(_config.Theta * (nodeOutput - synapse.Weight));
            });
        }

        private ISlaveNode _getWinner(IEnumerable<double> output)
        {
            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            return _network.OutputLayer.Nodes[winnerIndex] as ISlaveNode;
        }

        #endregion

    }
}