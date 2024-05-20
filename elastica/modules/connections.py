__doc__ = """
Connect
-------

Provides the connections interface to connect entities (rods,
rigid bodies) using joints (see `joints.py`).
"""
from typing import Type, Protocol, TypeAlias, cast
from typing_extensions import Self
from elastica.typing import (
    SystemIdxType,
    OperatorParam,
    OperatorFinalizeType,
    SystemType,
)
import numpy as np
from elastica.joint import FreeJoint

from .protocol import SystemCollectionProtocol, ModuleProtocol

ConnectionIndex: TypeAlias = (
    int | np.int_ | list[int] | tuple[int] | np.ndarray[int] | None
)


class SystemCollectionWithConnectionProtocol(SystemCollectionProtocol, Protocol):
    _connections: list[ModuleProtocol]

    _finalize_connections: OperatorFinalizeType

    def connect(
        self,
        first_rod: SystemType,
        second_rod: SystemType,
        first_connect_idx: int = None,
        second_connect_idx: int = None,
    ) -> ModuleProtocol: ...


class Connections:
    """
    The Connections class is a module for connecting rod-like objects using joints selected
    by the user. To connect two rod-like objects, the simulator class must be derived from
    the Connections class.

        Attributes
        ----------
        _connections: list
            List of joint classes defined for rod-like objects.
    """

    def __init__(self: SystemCollectionWithConnectionProtocol):
        self._connections = []
        super(Connections, self).__init__()
        self._feature_group_finalize.append(self._finalize_connections)

    def connect(
        self: SystemCollectionWithConnectionProtocol,
        first_rod: SystemType,
        second_rod: SystemType,
        first_connect_idx: ConnectionIndex = None,
        second_connect_idx: ConnectionIndex = None,
    ) -> ModuleProtocol:
        """
        This method connects two rod-like objects using the selected joint class.
        You need to input the two rod-like objects that are to be connected as well
        as set the element indexes of these rods where the connection occurs.

        Parameters
        ----------
        first_rod : SystemType
            Rod-like object
        second_rod : SystemType
            Rod-like object
        first_connect_idx : Optional[int]
            Index of first rod for joint.
        second_connect_idx : Optional[int]
            Index of second rod for joint.

        Returns
        -------

        """
        sys_idx = [None] * 2
        for i_sys, sys in enumerate((first_rod, second_rod)):
            sys_idx[i_sys] = self._get_sys_idx_if_valid(sys)

        # For each system identified, get max dofs
        # FIXME: Revert back to len, it should be able to take, systems without elements!
        # sys_dofs = [len(self._systems[idx]) for idx in sys_idx]
        sys_dofs = [self._systems[idx].n_elems for idx in sys_idx]

        # Create _Connect object, cache it and return to user
        _connect: ModuleProtocol = _Connect(*sys_idx, *sys_dofs)
        _connect.set_index(first_connect_idx, second_connect_idx)  # type: ignore[attr-defined]
        self._connections.append(_connect)
        self._feature_group_synchronize.append_id(_connect)

        return _connect

    def _finalize_connections(self: SystemCollectionWithConnectionProtocol) -> None:
        # From stored _Connect objects, instantiate the joints and store it
        # dev : the first indices stores the
        # (first rod index, second_rod_idx, connection_idx_on_first_rod, connection_idx_on_second_rod)
        # to apply the connections to.

        for connection in self._connections:
            first_sys_idx, second_sys_idx, first_connect_idx, second_connect_idx = (
                connection.id()
            )
            connect_instance: FreeJoint = connection.instantiate()

            # FIXME: lambda t is included because OperatorType takes time as an argument
            def apply_forces(time: np.floating) -> None:
                connect_instance.apply_forces(
                    system_one=self._systems[first_sys_idx],
                    index_one=first_connect_idx,
                    system_two=self._systems[second_sys_idx],
                    index_two=second_connect_idx,
                )

            def apply_torques(time: np.floating) -> None:
                connect_instance.apply_torques(
                    system_one=self._systems[first_sys_idx],
                    index_one=first_connect_idx,
                    system_two=self._systems[second_sys_idx],
                    index_two=second_connect_idx,
                )

            self._feature_group_synchronize.add_operators(
                connection, [apply_forces, apply_torques]
            )

        self._connections = []
        del self._connections

        # Need to finally solve CPP here, if we are doing things properly
        # This is to optimize the call tree for better memory accesses
        # https://brooksandrew.github.io/simpleblog/articles/intro-to-graph-optimization-solving-cpp/


class _Connect:
    """
    Connect module private class

    Attributes
    ----------
    _first_sys_idx: SystemIdxType
    _second_sys_idx: SystemIdxType
    _first_sys_n_lim: int
    _second_sys_n_lim: int
    _connect_class: list
    first_sys_connection_idx: int
    second_sys_connection_idx: int
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.
    """

    def __init__(
        self,
        first_sys_idx: SystemIdxType,
        second_sys_idx: SystemIdxType,
        first_sys_nlim: int,
        second_sys_nlim: int,
    ):
        """

        Parameters
        ----------
        first_sys_idx: int
        second_sys_idx: int
        first_sys_nlim: int
        second_sys_nlim: int
        """
        self._first_sys_idx: SystemIdxType = first_sys_idx
        self._second_sys_idx: SystemIdxType = second_sys_idx
        self._first_sys_n_lim: int = first_sys_nlim
        self._second_sys_n_lim: int = second_sys_nlim
        self._connect_cls: Type[FreeJoint] = None
        self.first_sys_connection_idx: ConnectionIndex = None
        self.second_sys_connection_idx: ConnectionIndex = None

    def set_index(
        self, first_idx: ConnectionIndex, second_idx: ConnectionIndex
    ) -> None:
        first_type = type(first_idx)
        second_type = type(second_idx)
        # Check if the types of first rod idx and second rod idx variable are same.
        assert (
            first_type == second_type
        ), f"Type of first_connect_idx :{first_type} is different than second_connect_idx :{second_type}"

        # Check if the type of idx variables are correct.
        allow_types = (int, np.int_, list, tuple, np.ndarray, type(None))
        assert isinstance(
            first_idx, allow_types
        ), f"Connection index type is not supported :{first_type}, please try one of the following :{allow_types}"

        # If type of idx variables are tuple or list or np.ndarray, check validity of each entry.
        if isinstance(first_idx, (tuple, list, np.ndarray)):
            first_idx_ = cast(list[int], first_idx)
            second_idx_ = cast(list[int], second_idx)
            for i in range(len(first_idx_)):
                assert isinstance(first_idx_[i], (int, np.int_)), (
                    "Connection index of first rod is not integer :{}".format(
                        first_idx_[i]
                    )
                    + " It should be : integer. Check your input!"
                )
                assert isinstance(second_idx_[i], (int, np.int_)), (
                    "Connection index of second rod is not integer :{}".format(
                        second_idx_[i]
                    )
                    + " It should be : integer. Check your input!"
                )

                # The addition of +1 and and <= check on the RHS is because
                # connections can be made to the node indices as well
                assert (
                    -(self._first_sys_n_lim + 1)
                    <= first_idx_[i]
                    <= self._first_sys_n_lim
                ), "Connection index of first rod exceeds its dof : {}".format(
                    self._first_sys_n_lim
                )
                assert (
                    -(self._second_sys_n_lim + 1)
                    <= second_idx_[i]
                    <= self._second_sys_n_lim
                ), "Connection index of second rod exceeds its dof : {}".format(
                    self._second_sys_n_lim
                )
        elif first_idx is None:
            # Do nothing if idx are None
            pass
        else:
            # The addition of +1 and and <= check on the RHS is because
            # connections can be made to the node indices as well
            first_idx__ = cast(int, first_idx)
            second_idx__ = cast(int, second_idx)
            assert (
                -(self._first_sys_n_lim + 1) <= first_idx__ <= self._first_sys_n_lim
            ), "Connection index of first rod exceeds its dof : {}".format(
                self._first_sys_n_lim
            )
            assert (
                -(self._second_sys_n_lim + 1) <= second_idx__ <= self._second_sys_n_lim
            ), "Connection index of second rod exceeds its dof : {}".format(
                self._second_sys_n_lim
            )

        self.first_sys_connection_idx = first_idx
        self.second_sys_connection_idx = second_idx

    def using(
        self,
        connect_cls: Type[FreeJoint],
        *args: OperatorParam.args,
        **kwargs: OperatorParam.kwargs,
    ) -> Self:
        """
        This method is a module to set which joint class is used to connect
        user defined rod-like objects.

        Parameters
        ----------
        connect_cls: object
            User defined callback class.
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        assert issubclass(
            connect_cls, FreeJoint
        ), "{} is not a valid joint class. Did you forget to derive from FreeJoint?".format(
            connect_cls
        )
        self._connect_cls = connect_cls
        self._args = args
        self._kwargs = kwargs
        return self

    def id(
        self,
    ) -> tuple[SystemIdxType, SystemIdxType, ConnectionIndex, ConnectionIndex]:
        return (
            self._first_sys_idx,
            self._second_sys_idx,
            self.first_sys_connection_idx,
            self.second_sys_connection_idx,
        )

    def instantiate(self) -> FreeJoint:
        if self._connect_cls is None:
            raise RuntimeError(
                "No connections provided to link rod id {0}"
                "(at {2}) and {1} (at {3}), but a Connection"
                "was intended as per code. Did you forget to"
                "call the `using` method?".format(*self.id())
            )

        try:
            return self._connect_cls(*self._args, **self._kwargs)
        except (TypeError, IndexError):
            raise TypeError(
                r"Unable to construct connection class.\n"
                r"Did you provide all necessary joint properties?"
            )
