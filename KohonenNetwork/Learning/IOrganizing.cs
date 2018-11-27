using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace KohonenNetwork.Learning
{

    [Obsolete]
    public interface IOrganizing
    {

        Task<bool> Organize(IEnumerable<double> input);

    }
}