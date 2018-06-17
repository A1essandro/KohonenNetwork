using KohonenNetwork;
using KohonenNetwork.Learning;
using NeuralNetworkConstructor.Network.Node.ActivationFunction;
using Xunit;
using System;

namespace Test
{
    public class LearningTest
    {

        [Fact]
        public void Learning()
        {
            var config = new NetworkConfiguration(3, 5);
            var network = new KohonenNetwork<Logistic>(config);

            network.SetLearning(new SelfLearning(0.1));
            var random = new Random();

            var control = new double[][]
            {
                new double[] {0.99, 0.49, 0.01},
                new double[] {0.03, 0.41, 0.99},
                new double[] {0.04, 0.55, 0.95}
            };

            for (var i = 0; i < 1000; i++)
            {
                var input = new double[] { random.NextDouble(), random.NextDouble(), random.NextDouble() };
                if (i % 10 == 0)
                {
                    Console.Write(Math.Round(input[0], 2) + " ");
                    Console.Write(Math.Round(input[1], 2) + " ");
                    Console.Write(Math.Round(input[2], 2) + " ");
                    network.Input(input);
                    Console.Write(" => " + network.GetOutputIndex() + Environment.NewLine);
                }
                network.Learn(input);
            }

            network.Input(control[0]);
            var res0 = network.GetOutputIndex();
            network.Input(control[1]);
            var res1 = network.GetOutputIndex();
            network.Input(control[2]);
            var res2 = network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

        [Fact]
        public void Organizing()
        {
            var config = new NetworkConfiguration(3, 1);
            var network = new KohonenNetwork<Logistic>(config);

            var organizing = new Organizing<Logistic>(network, 0.85);
            network.SetLearning(new SelfLearning(0.1, organizing));
            var random = new Random();

            var control = new double[][]
            {
                new double[] {0.99, 0.49, 0.01},
                new double[] {0.03, 0.41, 0.99},
                new double[] {0.04, 0.55, 0.95}
            };

            for (var i = 0; i < 1000; i++)
            {
                var input = new double[] { random.NextDouble(), random.NextDouble(), random.NextDouble() };
                network.Learn(input);
            }

            network.Input(control[0]);
            var res0 = network.GetOutputIndex();
            network.Input(control[1]);
            var res1 = network.GetOutputIndex();
            network.Input(control[2]);
            var res2 = network.GetOutputIndex();

            Assert.NotEqual(res0, res1);
            Assert.Equal(res1, res2);
        }

    }
}