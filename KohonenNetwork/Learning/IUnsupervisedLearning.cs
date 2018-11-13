using System.Collections.Generic;
using System.Threading.Tasks;

namespace KohonenNetwork.Learning
{
    public interface IUnsupervisedLearning
    {

        Task Learn(IEnumerable<double> input);

        Task Learn(IEnumerable<IEnumerable<double>> epoch, int? repeats = null);

    }
}