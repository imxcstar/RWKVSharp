using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RWKVSharp.Core.Interfaces
{
    public interface IRunnerFactoryBuilder
    {
        public IRunner Builder();
    }
}
