using KohonenNetwork;
using KohonenNetwork.Learning;
using Xunit;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Structure.ActivationFunctions;
using NeuralNetworkConstructor.Structure.Layers;
using NeuralNetworkConstructor.Structure.Nodes;
using NeuralNetworkConstructor.Learning;
using NeuralNetworkConstructor.Learning.Samples;
using NeuralNetworkConstructor.Constructor.Generators;
using NeuralNetworkConstructor.Structure.Synapses;
using KNetwork = KohonenNetwork.KohonenNetwork;

namespace Test
{
    public class LearningTest
    {

        private double[][] _control = new double[][]
            {
                new double[] {0.99, 0.49, 0.01},
                new double[] {0.03, 0.47, 0.99},
                new double[] {0.04, 0.55, 0.96}
            };

        private double[][] _getInputs()
        {
            var random = new Random();

            var inputs = new double[2500][];
            for (var i = 0; i < 2500; i++)
            {
                inputs[i] = new double[] { random.NextDouble(), random.NextDouble(), random.NextDouble() };
            }

            return inputs;
        }

        [Fact]
        public async Task Learning1()
        {
            var inputLayer = new InputLayer(() => new InputNode(), 3);
            var outputLayer = new Layer(() => new Neuron(), 5);

            new EachToEachSynapseGenerator<Synapse>().Generate(inputLayer, outputLayer);

            var network = new KNetwork(inputLayer, outputLayer);
            var strategy = new KohonenNetwork.Learning.Strategy.UnsupervisedLearning();
            var learning = new Learning<KNetwork, ISelfLearningSample>(network, strategy, new LearningSettings
            {
                EpochRepeats = 100,
                ThetaFactorPerEpoch = x => 0.975
            });

            var inputs = _getInputs().Select(x => new SelfLearningSample(x));
            await learning.Learn(inputs);

            network.Input(_control[0]);
            var res0 = await network.GetOutputIndex();
            network.Input(_control[1]);
            var res1 = await network.GetOutputIndex();
            network.Input(_control[2]);
            var res2 = await network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

    }
}