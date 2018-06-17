using KohonenNetwork;
using KohonenNetwork.Learning;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using Xunit;
using System;

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
        public void Learning()
        {
            var networkConfig = new NetworkConfiguration(3, 5);
            var learningConfig = new LearningConfiguration
            {
                ThetaFactorPerEpoch = 0.95
            };
            var network = new KohonenNetwork<Logistic>(networkConfig);
            var learning = new UnsupervisedLearning(network, learningConfig);

            var inputs = _getInputs();
            learning.Learn(inputs, 25);

            network.Input(_control[0]);
            var res0 = network.GetOutputIndex();
            network.Input(_control[1]);
            var res1 = network.GetOutputIndex();
            network.Input(_control[2]);
            var res2 = network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

        [Fact]
        public void Organizing()
        {
            var networkConfig = new NetworkConfiguration(3, 1);
            var network = new KohonenNetwork<Logistic>(networkConfig);
            var learningConfig = new LearningConfiguration
            {
                ThetaFactorPerEpoch = 0.95,
                OrganizingAlgorithm = new Organizing<Logistic>(network, 0.777)
            };
            var learning = new UnsupervisedLearning(network, learningConfig);

            var inputs = _getInputs();
            learning.Learn(inputs, 25);

            network.Input(_control[0]);
            var res0 = network.GetOutputIndex();
            network.Input(_control[1]);
            var res1 = network.GetOutputIndex();
            network.Input(_control[2]);
            var res2 = network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

    }
}