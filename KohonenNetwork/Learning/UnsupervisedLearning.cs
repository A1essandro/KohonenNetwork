using System;
using System.Collections.Generic;
using System.Linq;
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
            var output = _network.Output();

            var winnerIndex = Array.IndexOf(output.ToArray(), output.Max());
            var winner = _network.OutputLayer.Nodes[winnerIndex] as ISlaveNode;

            foreach (var synapse in winner.Synapses)
            {
                var nodeOutput = synapse.MasterNode.Output();
                synapse.ChangeWeight(_config.Theta * (nodeOutput - synapse.Weight));
            }
        }

        public void Learn(IEnumerable<IEnumerable<double>> epoch, int repeats = 1)
        {
            var random = new Random();
            var initialTheta = _config.Theta;

            for (var i = 0; i < repeats; i++)
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

        public void SetNetwork(INetwork network)
        {
            _network = network;
        }

    }
}