using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RWKVSharp.Core.Interfaces;
using RWKVSharp.Core.Sampler;
using RWKVSharp.Core.Tokenizer;

namespace RWKVSharp.Core
{
    public class RunnerOptions : IRunnerOptions
    {
        public Func<ITokenizer> Tokenizer { get; set; } = null!;
        public Func<ISampler> Sampler { get; set; } = null!;
    }
}
