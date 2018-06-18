using System.Collections.Generic;
using System.Threading.Tasks;
using NeuralNetworkConstructor.Network;

namespace KohonenNetwork.Learning
{
    public interface IUnsupervisedLearning
    {

        void Learn(IEnumerable<double> input);

        Task LearnAsync(IEnumerable<double> input);

        void Learn(IEnumerable<IEnumerable<double>> epoch, int repeats = 1);

        Task LearnAsync(IEnumerable<IEnumerable<double>> epoch, int repeats = 1);

    }
}