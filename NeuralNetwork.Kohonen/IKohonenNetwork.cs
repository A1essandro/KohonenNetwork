using NeuralNetwork.Structure.Networks;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NeuralNetwork.Kohonen
{
    public interface IKohonenNetwork : INetwork
    {

        Task<IEnumerable<double>> RawOutput();

        Task<int?> GetOutputIndex();

    }
}