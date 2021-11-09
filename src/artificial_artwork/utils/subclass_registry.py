"""Exposes the SubclassRegistry that allows to define a single registration point of one or more subclasses of a
(common parent) class."""


class SubclassRegistry(type):
    """Subclass Registry

    A (parent) class using this class as metaclass gains the 'subclasses' class attribute as well as the 'create' and
    'register_as_subclass' class methods.

    The 'subclasses' attribute is a python dictionary having string identifiers as keys and subclasses of the (parent)
    class as values.

    The 'register_as_subclass' class method can be used as a decorator to indicate that a (child) class should belong in
    the parent's class registry. An input string argument will be used as the unique key to register the subclass.

    The 'create' class method can be invoked with a (string) key and suitable constructor arguments to later construct
    instances of the corresponding child class.

    Example:

        >>> from so_magic.utils import SubclassRegistry

        >>> class ParentClass(metaclass=SubclassRegistry):
        ...  pass

        >>> ParentClass.subclasses
        {}

        >>> @ParentClass.register_as_subclass('child')
        ... class ChildClass(ParentClass):
        ...  def __init__(self, child_attribute):
        ...   self.attr = child_attribute

        >>> child_instance = ParentClass.create('child', 'attribute-value')
        >>> child_instance.attr
        'attribute-value'

        >>> type(child_instance).__name__
        'ChildClass'

        >>> isinstance(child_instance, ChildClass)
        True

        >>> isinstance(child_instance, ParentClass)
        True

        >>> {k: v.__name__ for k, v in ParentClass.subclasses.items()}
        {'child': 'ChildClass'}
    """
    def __new__(mcs, *args, **kwargs):
        class_object = super().__new__(mcs, *args, **kwargs)
        class_object.subclasses = {}
        return class_object

    def create(cls, subclass_identifier, *args, **kwargs):
        """Create an instance of a registered subclass, given its unique identifier and runtime (constructor) arguments.

        Invokes the identified subclass constructor passing any supplied arguments. The user needs to know the arguments
        to supply depending on the resulting constructor signature.

        Args:
            subclass_identifier (str): the unique identifier under which to look for the corresponding subclass

        Raises:
            ValueError: In case the given identifier is unknown to the parent class
            InstantiationError: In case the runtime args and kwargs do not match the constructor signature

        Returns:
            object: the instance of the registered subclass
        """
        if subclass_identifier not in cls.subclasses:
            raise ValueError(f'Bad "{str(cls.__name__)}" subclass request; requested subclass with identifier '
                             f'{str(subclass_identifier)}, but known identifiers are '
                             f'[{", ".join(str(subclass_id) for subclass_id in cls.subclasses.keys())}]')
        try:
            return cls.subclasses[subclass_identifier](*args, **kwargs)
        except Exception as any_error:
            raise InstantiationError('Error during instance object construction.'
                f' Failed to create instance of class {cls.subclasses[subclass_identifier]}'
                f' using args [{", ".join((str(_) for _ in args))}]'
                f' and kwargs [{", ".join(f"{k}={v}" for k, v in kwargs.items())}]') from any_error

    def register_as_subclass(cls, subclass_identifier):
        """Register a class as subclass of the parent class.

        Adds the subclass' constructor in the registry (dict) under the given (str) identifier. Overrides the registry
        in case of "identifier collision". Can be used as a python decorator.

        Args:
            subclass_identifier (str): the user-defined identifier, under which to register the subclass
        """
        def wrapper(subclass):
            """Add the (sub) class provided to the parent class registry.

            Args:
                subclass ([type]): the (sub) class to register

            Returns:
                object: the (sub) class
            """
            cls.subclasses[subclass_identifier] = subclass
            return subclass
        return wrapper


class InstantiationError(Exception): pass
