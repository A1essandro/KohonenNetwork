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
            var config = new NetworkConfiguration(3, 5);
            var network = new KohonenNetwork<Logistic>(config);
            network.SetLearning(new SelfLearning(0.075));

            var inputs = _getInputs();
            network.Learn(inputs, 25);

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
            var config = new NetworkConfiguration(3, 1);
            var network = new KohonenNetwork<Logistic>(config);
            var organizing = new Organizing<Logistic>(network, 0.777);
            network.SetLearning(new SelfLearning(0.075, organizing));

            var inputs = _getInputs();
            network.Learn(inputs, 25);

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