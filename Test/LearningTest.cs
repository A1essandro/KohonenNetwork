using KohonenNetwork;
using KohonenNetwork.Learning;
using Xunit;
using System;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Structure.ActivationFunctions;

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
        public async Task Learning()
        {
            var networkConfig = new NetworkConfiguration(3, 5);
            var learningConfig = new LearningConfiguration
            {
                ThetaFactorPerEpoch = 0.95,
                DefaultRepeatsNumber = 25
            };
            var network = new KohonenNetwork.KohonenNetwork(networkConfig);
            var learning = new UnsupervisedLearning(network, learningConfig);

            var inputs = _getInputs();
            await learning.Learn(inputs);

            await network.Input(_control[0]);
            var res0 = network.GetOutputIndex();
            await network.Input(_control[1]);
            var res1 = network.GetOutputIndex();
            await network.Input(_control[2]);
            var res2 = network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

        [Fact]
        public async Task Organizing()
        {
            var networkConfig = new NetworkConfiguration(3, 1);
            var network = new KohonenNetwork.KohonenNetwork(networkConfig);
            var learningConfig = new LearningConfiguration
            {
                ThetaFactorPerEpoch = 0.95,
                OrganizingAlgorithm = new Organizing(network, 0.777)
            };
            var learning = new UnsupervisedLearning(network, learningConfig);

            var inputs = _getInputs();
            await learning.Learn(inputs, 25);

            await network.Input(_control[0]);
            var res0 = await network.GetOutputIndex();
            await network.Input(_control[1]);
            var res1 = await network.GetOutputIndex();
            await network.Input(_control[2]);
            var res2 = await network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

    }
}