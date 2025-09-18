"""
Dependency injection container for the Pele processing system.

Manages service registration, resolution, and lifecycle to enable
loose coupling and testability.
"""
from typing import Dict, Type, Any, Optional, Callable, TypeVar, cast
from functools import wraps
import threading
from enum import Enum

from .exceptions import DependencyError, ConfigurationError

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime management options."""
    SINGLETON = "singleton"  # Single instance shared across application
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # Single instance per scope (e.g., per request)


class ServiceDescriptor:
    """Describes how a service should be created and managed."""

    def __init__(self,
                 service_type: Type,
                 implementation: Optional[Any] = None,
                 factory: Optional[Callable] = None,
                 lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
                 dependencies: Optional[Dict[str, Type]] = None):
        self.service_type = service_type
        self.implementation = implementation
        self.factory = factory
        self.lifetime = lifetime
        self.dependencies = dependencies or {}
        self._instance = None
        self._lock = threading.Lock()

        if not implementation and not factory:
            raise ConfigurationError(
                "service_registration",
                f"Must provide either implementation or factory for {service_type.__name__}"
            )

    def create_instance(self, container: 'Container') -> Any:
        """Create service instance based on descriptor configuration."""
        if self.lifetime == ServiceLifetime.SINGLETON:
            if self._instance is None:
                with self._lock:
                    if self._instance is None:
                        self._instance = self._create_new_instance(container)
            return self._instance
        else:
            return self._create_new_instance(container)

    def _create_new_instance(self, container: 'Container') -> Any:
        """Create new instance using factory or implementation."""
        if self.factory:
            # Resolve factory dependencies
            resolved_deps = {}
            for param_name, dep_type in self.dependencies.items():
                resolved_deps[param_name] = container.resolve(dep_type)
            return self.factory(**resolved_deps)
        else:
            return self.implementation


class Container:
    """Dependency injection container."""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._lock = threading.Lock()

    def register_singleton(self, service_type: Type[T],
                           implementation: Optional[T] = None,
                           factory: Optional[Callable[..., T]] = None,
                           **dependencies) -> 'Container':
        """Register a singleton service.

        Args:
            service_type: Interface or abstract class type
            implementation: Concrete instance (if pre-created)
            factory: Factory function to create instance
            **dependencies: Named dependencies for factory function

        Returns:
            Self for method chaining
        """
        return self._register_service(
            service_type, implementation, factory,
            ServiceLifetime.SINGLETON, dependencies
        )

    def register_transient(self, service_type: Type[T],
                           implementation: Optional[T] = None,
                           factory: Optional[Callable[..., T]] = None,
                           **dependencies) -> 'Container':
        """Register a transient service (new instance each time).

        Args:
            service_type: Interface or abstract class type
            implementation: Concrete instance template
            factory: Factory function to create instances
            **dependencies: Named dependencies for factory function

        Returns:
            Self for method chaining
        """
        return self._register_service(
            service_type, implementation, factory,
            ServiceLifetime.TRANSIENT, dependencies
        )

    def register_instance(self, service_type: Type[T], instance: T) -> 'Container':
        """Register a pre-created instance as singleton.

        Args:
            service_type: Service interface type
            instance: Pre-created instance

        Returns:
            Self for method chaining
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        descriptor._instance = instance

        with self._lock:
            self._services[service_type] = descriptor

        return self

    def _register_service(self, service_type: Type, implementation: Optional[Any],
                          factory: Optional[Callable], lifetime: ServiceLifetime,
                          dependencies: Dict[str, Type]) -> 'Container':
        """Internal service registration method."""
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            factory=factory,
            lifetime=lifetime,
            dependencies=dependencies
        )

        with self._lock:
            self._services[service_type] = descriptor

        return self

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance.

        Args:
            service_type: Type of service to resolve

        Returns:
            Service instance

        Raises:
            DependencyError: If service is not registered
        """
        if service_type not in self._services:
            raise DependencyError(
                service_type.__name__,
                required_version="registered",
                found_version="not_found"
            )

        descriptor = self._services[service_type]
        return cast(T, descriptor.create_instance(self))

    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not found.

        Args:
            service_type: Type of service to resolve

        Returns:
            Service instance or None if not registered
        """
        try:
            return self.resolve(service_type)
        except DependencyError:
            return None

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered.

        Args:
            service_type: Type to check

        Returns:
            True if service is registered
        """
        return service_type in self._services

    def get_registered_services(self) -> Dict[Type, ServiceLifetime]:
        """Get all registered services and their lifetimes.

        Returns:
            Dictionary mapping service types to lifetimes
        """
        return {
            service_type: descriptor.lifetime
            for service_type, descriptor in self._services.items()
        }

    def clear(self) -> None:
        """Clear all registered services."""
        with self._lock:
            # Clear singleton instances
            for descriptor in self._services.values():
                descriptor._instance = None
            self._services.clear()

    def create_child_container(self) -> 'Container':
        """Create child container that inherits parent registrations.

        Returns:
            Child container
        """
        child = Container()
        child._services = self._services.copy()
        return child


def inject(*dependencies: Type) -> Callable:
    """Decorator to inject dependencies into function parameters.

    Args:
        *dependencies: Types to inject in parameter order

    Returns:
        Decorated function with dependency injection
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get container from first argument or global
            container = None
            if args and hasattr(args[0], '_container'):
                container = args[0]._container
            else:
                container = _global_container

            if not container:
                raise DependencyError(
                    "container",
                    required_version="available",
                    found_version="not_configured"
                )

            # Resolve dependencies and add to kwargs
            for i, dep_type in enumerate(dependencies):
                if len(args) <= i + 1:  # Skip already provided args
                    param_name = func.__code__.co_varnames[i + 1]  # Skip 'self'
                    if param_name not in kwargs:
                        kwargs[param_name] = container.resolve(dep_type)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Global container for simple scenarios
_global_container: Optional[Container] = None


def get_global_container() -> Container:
    """Get or create global container instance.

    Returns:
        Global container instance
    """
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def set_global_container(container: Container) -> None:
    """Set the global container instance.

    Args:
        container: Container to use globally
    """
    global _global_container
    _global_container = container


def configure_default_services(container: Container) -> None:
    """Configure default service registrations.

    Args:
        container: Container to configure
    """
    # This would be implemented based on available concrete classes
    # For now, it's a placeholder for the configuration logic
    pass


class ServiceRegistry:
    """Helper class for organizing service registrations."""

    def __init__(self, container: Container):
        self.container = container

    def register_data_services(self, **config) -> None:
        """Register data layer services."""
        # Would register DataLoader, DataExtractor, etc.
        # Implementation depends on available concrete classes
        pass

    def register_analysis_services(self, **config) -> None:
        """Register analysis layer services."""
        # Would register FlameAnalyzer, ShockAnalyzer, etc.
        pass

    def register_visualization_services(self, **config) -> None:
        """Register visualization services."""
        # Would register FrameGenerator, AnimationBuilder, etc.
        pass

    def register_parallel_services(self, **config) -> None:
        """Register parallel processing services."""
        # Would register ParallelCoordinator, WorkDistributor, etc.
        pass

    def register_utility_services(self, **config) -> None:
        """Register utility services."""
        # Would register Logger, UnitConverter, etc.
        pass


# Context manager for scoped services
class ServiceScope:
    """Context manager for scoped service lifetime."""

    def __init__(self, container: Container):
        self.container = container
        self.scoped_container = container.create_child_container()

    def __enter__(self) -> Container:
        return self.scoped_container

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up scoped services
        for service_type, descriptor in self.scoped_container._services.items():
            if descriptor.lifetime == ServiceLifetime.SCOPED:
                descriptor._instance = None