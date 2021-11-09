import pytest

@pytest.fixture
def proxy_module():
    from artificial_artwork.utils import proxy
    return proxy


@pytest.fixture
def dummy_handle():
    def handle(self, *args, **kwargs):
        return f'{type(self).__name__} handle request with args [{", ".join((str(_) for _ in args))}] and kwargs [{", ".join(f"{k}={v}" for k, v in kwargs.items())}]'
    return handle


def test_proxy_behaviour(proxy_module, dummy_handle, capsys):
    prm = proxy_module

    # replicate client code that wants to use the proxy pattern

    # Derive from class RealSubject or use 'python overloading' (use a class 
    # that has a 'request' method with the same signature as ReadSubject.request)
    class ClientSubject(prm.RealSubject):
        def request(self, *args, **kwargs):
            # delegate handling to the real real subject from client code
            # so frankly the request method here is also an adapter
            print(dummy_handle(self, *args, **kwargs))
            return type(self).__name__

    # Derive from Proxy
    class ClientProxy(prm.Proxy):
        
        def request(self, *args, **kwargs):
            
            # run proxy code before sending request to the "proxied" handler
            before_args = list(['before'] + list(args))
            print(dummy_handle(self, *before_args, **kwargs))
            
            # handle request with the proxied logic
            _ = super().request(*args, **kwargs)
            assert _ == 'ClientSubject'

            # run proxy code after request to the "proxied" handler
            after_args = list(['after'] + list(args))
            print(dummy_handle(self, *after_args, **kwargs))
            return _

    real_subject = ClientSubject()
    proxy = ClientProxy(real_subject)

    # use proxy in a scenario

    # First test what happens without using proxy
    args = [1, 2]
    kwargs = {'k1': 'v1'}
    result = real_subject.request(*args, **kwargs)

    captured = capsys.readouterr()
    assert captured.out == dummy_handle(real_subject, 1, 2, k1='v1') + '\n'
    assert captured.out == f'ClientSubject handle request with args [{", ".join(str(_) for _ in args)}] and kwargs [{", ".join(f"{k}={v}" for k, v in kwargs.items())}]\n'
    assert result == type(real_subject).__name__
    assert result == 'ClientSubject'

    # Now test what happens using proxy
    result = proxy.request(*args, **kwargs)

    captured = capsys.readouterr()
    assert captured.out == dummy_handle(*list([proxy, 'before'] + args), **kwargs) +'\n' + \
        dummy_handle(*list([real_subject] + args), **kwargs) + '\n' + \
        dummy_handle(*list([proxy, 'after'] + args), **kwargs) +'\n'
    assert captured.out == f'ClientProxy handle request with args [{", ".join(str(_) for _ in ["before"] + args)}] and kwargs [{", ".join(f"{k}={v}" for k, v in kwargs.items())}]\n' + \
        f'ClientSubject handle request with args [{", ".join(str(_) for _ in args)}] and kwargs [{", ".join(f"{k}={v}" for k, v in kwargs.items())}]\n' + \
        f'ClientProxy handle request with args [{", ".join(str(_) for _ in ["after"] + args)}] and kwargs [{", ".join(f"{k}={v}" for k, v in kwargs.items())}]\n'
    assert result == type(real_subject).__name__
    assert result == 'ClientSubject'
